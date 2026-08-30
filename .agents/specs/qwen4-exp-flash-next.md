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

### W5b-4 correction: the indexer's SCORE and TOP-K are ALREADY `vt::` ops

Issue [#2167](https://github.com/mudler/vllm.cpp/issues/2167) opened with a
"why nothing existing serves it" table naming `IndexSelect`, `TopKValuesIndices`,
`GatherMlaCache` and the fused `kDeepseekV4Dsa` / `kDeepseekV4Compressor`. **That
table is incomplete, and the two ops it omits are the two that do serve.** The
correction is recorded here rather than in the issue, which is append-only in
practice, because a later wave reading the issue would otherwise re-derive it.

`vt::DsaIndexerLogits` (`include/vt/ops.h`, kernel `src/vt/cpu/cpu_dsa_indexer.cpp`)
computes, over a ONE-key-head MQA cache with a per-query `[win_start, win_end)`
window:

```
logit[t,s] = sum_h fold[t,h] * ReLU(dot(q[t,h,:], k[s,:]))
fold[t,h]  = weights[t,h] * q_scale[t,h] * softmax_scale * n_head_scale
```

With `weights` all ones, `q_scale` null and `n_head_scale = 1`, the fold
collapses to the single constant `softmax_scale`. Set that to
`index_head_dim ** -0.5` and this **is** `Qwen4ExpTextQSAIndexer`'s block score
(`modeling_qwen4_exp.py:690-693`): QSA has neither DeepSeek-V4's learned
`weights_proj` nor its `n_head ** -0.5`, so the constant is all that is left.
`vt::DsaTopkSelect` is the same all-select-below-k, ties-to-the-LOWER-index,
ASCENDING-emission top-k — the exact three semantics `QsaTopkBlocks` inherited
from `sampler.cu` in W4 — applied to the block axis instead of the token axis.

So the QSA indexer is `Qwen4ExpQsaCompress` followed by those two, with
`win_start[t] = 0` and `win_end[t] = kv_len[t] / compress_ratio`. Adding a
QSA-private scoring kernel beside `cpu_dsa_indexer.cpp` would have been the
parallel path AGENTS.md §"Shared seams" forbids, in the same file that already
declines to re-implement `k_norm` and the leading-slice rope for precisely that
reason.

**One reassociation survives, and it is named rather than hidden.** Upstream
divides AFTER the head sum and the fold multiplies BEFORE it — `c * sum_h r_h`
against `sum_h c * r_h`. Equal in exact arithmetic, up to an ulp apart in f32,
and top-k is invariant under a positive scalar, so no selection can move except
through a tie manufactured at that ulp. That is an argument, not a measurement,
which is why `tests/vllm/models/test_qwen4_exp_qsa_device.cpp` compares the
COMPOSED selection against the lane-pinned oracle's own selected token sets for
every query token of both fixtures, ragged tail included, rather than relying on
it. Mutation M27 is the paired control: making `weights` non-uniform breaks the
collapse and reds the suite, so the ones are load-bearing rather than decorative.
M26 is the other half — inheriting DeepSeek-V4's `n_head_scale` SURVIVES, which
is the same positive-rescale blindness W4 already recorded for `QsaBlockScores`,
and it is in the table so the pair reads as an instrument that is wired up.

**What remains genuinely new is two ops.** The mean pool has no `vt::`
counterpart at all — this tree has no mean, no pool, no axis reduction and no
transpose, so the non-overlapping window cannot even be faked as a strided
depthwise conv — and fusing it with the norm and the block-start rope mirrors
upstream, whose compressor is one kernel, on the in-tree `kFusedNormRope`
precedent. The GATHER consumer has no counterpart anywhere: every DeepSeek-V4
sparse consumer attends the COMPRESSED MLA KV and MiniMax-M3's attend raw tokens
at KV-PAGE granularity, while QSA attends RAW tokens selected at ratio-4
granularity.

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

### The KV-cache spec is THREE groups and ONE uniform recurrent group (W5c, #2031)

`MakeQwen4ExpKVCache` returns instead of refusing. Landed by W5c; the shape and
the reason are recorded here because the alternative shape is the one a reader
arrives with.

| # | layer_names | spec |
|---|---|---|
| 0 | the 12 QSA layers, `model.layers.<l>.self_attn.attn` | `FullAttentionSpec(block, 2, 256, ResolveKvCacheDType())` |
| 1 | the 36 linear layers, `model.layers.<l>.linear_attn` | `MambaSpec(block, {{10240,3},{48,128,128},{10240,9},{2}}, {bf16, ssm, bf16, kI64})` |
| 2 | the 12 QSA layers, `model.layers.<l>.self_attn.indexer.k_cache` | `MLAAttentionSpec(block, 128, ResolveKvCacheDType(), 1, …, compress_ratio=4)` |

**ONE uniform recurrent group, not per-layer specs and not several groups, and
that is the MIRROR rather than a shortcut.** Only ONE of the 36 linear layers
carries the PLE conv and the n-gram history, so a per-layer spec set is the
shape a reader expects. Upstream cannot produce it. Read at the pin
`5559679229`:

- `vllm/model_executor/models/interfaces.py:809-812` —
  `get_mamba_state_shape_from_config(cls, vllm_config)` is a CLASSMETHOD over
  the CONFIG, with no `layer_idx`. 19 definitions of that name tree-wide: this
  protocol declaration plus **18 implementations**, and not one of them takes a
  layer index.
- `vllm/v1/worker/mamba_utils.py:441` — `get_mamba_groups` asserts
  `all(mamba_specs[0] == spec for spec in mamba_specs)`: every `MambaSpec` in
  the model EQUAL, `shapes` and `dtypes` included.
- `vllm/v1/core/kv_cache_utils.py:1101-1109` — a `MambaSpec` whose page is
  smaller than the model's max is given `page_size_padded=max_page_size` and is
  otherwise unchanged. Upstream PADS. It does not split.

**The cost, derived and not measured.** The PLE conv is `10240 x 9` at bf16 =
184320 B and the n-gram history is 2 `int64` = 16 B, so **184336 B per sequence**
on each of the **35** linear layers that never read them: **49.2 MiB** at the
default `max_num_seqs` of 8, 0.09% of the GB10 headroom `## Hardware` accounts.
Gated as a literal in `tests/vllm/models/test_qwen4_exp_kv_cache.cpp` against
the same config with `ple_layer_ids` erased, so the number moves if the shapes
do. **No device has allocated it** — see `## Owed`.

This CORRECTS `.agents/specs/recurrent-multistate.md`, whose `## Owed` said a
second recurrent group "IS on a PLE topology's path". That measurement fed
upstream's grouping functions a heterogeneous per-layer input upstream never
constructs. Both halves stay owed as generic engine debt —
`ComputeHybridKvBudget` reads only the first mamba group
(`src/vllm/v1/core/hybrid_kv_budget.cpp:26`) — and neither is on this row's path.

**State order is `[gdn_conv, temporal, ple_conv, ngram]`, a deliberate
divergence from upstream's list order, and the same bytes.** Upstream keeps the
three CONV states adjacent (`number_of_conv_states = 3`) with the temporal state
after them. `GdnStateCache` publishes `conv_state = states[0]` and
`ssm_state = states[1]` as NAMED fields that THREE model families read —
`qwen3_5.cpp`, `kimi_linear_device.cpp`, and the `nemotron_h` pair
`nemotron_h_device.cpp` / `nemotron_h_forward.h` — so moving the temporal state
off slot 1 would silently re-point three model families. Slice order differs;
`page_size_bytes` does not.

**Three, not four ([#2203](https://github.com/mudler/vllm.cpp/issues/2203)).**
This wave first wrote FOUR here and in `qwen4_exp_registry.cpp`, inheriting the
list from `.agents/specs/recurrent-multistate.md` (landed by `f7710c1b4`,
[#2131](https://github.com/mudler/vllm.cpp/issues/2131)), whose fourth name is
`gemma4_mm.cpp`. That file reads NEITHER field — zero occurrences of
`conv_state` and zero of `ssm_state` — and its only two mentions of the type are
an include comment and `std::vector<GdnStateCache> no_gdn_state;`
(`gemma4_mm.cpp:221`), passed EMPTY. It is the file that proves Gemma-4 has no
recurrent arm, cited as proving the opposite. `muse_glimmer_mm.cpp:340` and
`qwen3_vl.cpp:621` carry the identical empty-vector shape, so the wrong fourth
name was one of the three files that demonstrate the negative. Measured on
`ad6696fa3`, `GdnStateCache` / `conv_state` / `ssm_state` counts per file:

| File | `GdnStateCache` | `conv_state` | `ssm_state` |
|---|---|---|---|
| `qwen3_5.cpp` | 37 | 33 | 34 |
| `nemotron_h_device.cpp` | 6 | 9 | 14 |
| `kimi_linear_device.cpp` | 2 | 7 | 6 |
| `gemma4_mm.cpp` | 2 | **0** | **0** |
| `muse_glimmer_mm.cpp` | 2 | **0** | **0** |
| `qwen3_vl.cpp` | 2 | **0** | **0** |

A grep on the FIELD name over-counts in the other direction:
`glm5_next_kda.cpp:343-345` matches `conv_state` 13 times, but that is
`Glm5NextKdaCache::conv_state`, a `std::vector<float>` KDA sequence state
(`include/vllm/model_executor/models/glm5_next_kda.h:314`), where this one is a
`vt::Tensor` (`include/vllm/model_executor/models/qwen3_5.h:111`); that file has
zero occurrences of `GdnStateCache`. Grep the TYPE. **The conclusion does not
move:** re-pointing three families is still the reason the temporal state stays
on slot 1. Only the enumeration was wrong, and no checker can see this class —
`check-symbol-anchors` resolves symbols, and `GdnStateCache` genuinely appears
in `gemma4_mm.cpp`, so symbol existence passes on a file whose behaviour is the
opposite of the one asserted. Same class as
[#2198](https://github.com/mudler/vllm.cpp/issues/2198), which this wave closes.

**Group 2 must be an `MLAAttentionSpec`, and a `FullAttentionSpec` there fails
SILENTLY.** The runner's leftover scan treats the first published
`kFullAttention` group that is neither the target nor the recurrent one as the
single `fa_draft` draft-KV slot and `continue`s
(`src/vllm/v1/worker/gpu/runner.cpp`, the `draft_slot_taken` arm). The leftover
count then stays 0, `multi_cache_topology` stays false, the legacy one-buffer-
per-layer path runs, and the side cache is published and never allocated with
nothing reported. `kMlaAttention` is not absorbed by that arm, so the topology
is multi-cache and every published cache gets a buffer.

**Real per-layer names, never placeholders.**
`ResolveKVCacheGroupLayerNames` rewrites a placeholder group set, but its
fallback can name only a TARGET attention group and one `fa_draft` slot: a THIRD
attention group reaches `group.layer_names.clear()`
(`src/vllm/v1/kv_cache_interface.cpp`), and an unnamed group is then refused by
the runner's multi-cache admission check for names that "do not all resolve to
distinct in-range layer indices". Publishing real names also makes that rewrite
a no-op by its own idempotence guard, gated in the KV suite.

**`block_size % compress_ratio != 0` is refused BY NAME**, because
`storage_block_size()` is integer division
(`vllm/v1/kv_cache_interface.py:393-395`) and truncating it sizes the page for
fewer states than the block covers — a short cache, i.e. wrong tokens rather
than a crash. Upstream never meets it (its DeepSeek-V4 block sizes are powers of
two above the ratio); ours arrives as a caller-supplied parameter.

**A non-uniform `compress_ratio` was ALREADY refused and was NOT gated.**
`Qwen4ExpHfConfigFromGguf` takes the first non-zero entry of the per-layer
`attention.compress_ratios` and requires the rest to agree — the mirror of
upstream's `MLAAttentionSpec.merge` assert
(`vllm/v1/kv_cache_interface.py:424-435`) — and deleting that `VT_CHECK` left
`test_qwen4_exp_gguf_weights` fully green. W5c gates it rather than adding a
second copy in the KV builder. The fixture has to DOUBLE `block_count` to reach
it: at four layers and `full_attention_interval` 4 there is exactly one sparse
layer, one non-zero ratio cannot disagree with itself, and a stray non-zero on a
linear layer is caught one check earlier by the schedule agreement.

**Two refusals W5c did NOT add, because W1 already has them**, verified rather
than assumed: a `ple_layer_ids` entry outside the one-indexed range, and a PLE
id landing on a layer the rewrite made sparse
(`src/vllm/model_executor/models/qwen4_exp.cpp`), both gated by named subcases
in `test_qwen4_exp_scaffold.cpp`. A second copy in the KV builder would be a
second derivation of one rule.

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
| `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S | **67.56 GiB**, 3 shards | **YES, and it is the ONLY published artifact that does** |

**CORRECTED 2026-08-26.** This table previously read "README only, zero weight files
-- does not exist", and that was true when it was written and false a few hours later.
The repository was populated at 13:32Z with `UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-0000{1,2,3}-of-00003.gguf`,
72,546,461,344 bytes = **67.56 GiB**, read from the files' own headers:
`general.architecture = "qwen4exp"`, `split.tensors.count = 1224`,
`general.file_type = 24`. **It fits GB10 with roughly 52 GiB of headroom**, where every
safetensors artifact does not fit at all. The whole shape of this row's `## Work
breakdown` follows from that, which is why the correction is called out rather than
quietly applied.

Its metadata independently confirms this spec's own n-gram derivation to the digit:
`qwen4exp.ple.layer_multipliers = [23703573157769, 20109073645365, 8052911324071]`,
`qwen4exp.ple.head_vocab_sizes` starting `[20000003, 20000023, 20000033, ...]`, and
`qwen4exp.ple.layers = [1]` (0-based) corroborating the one-indexed conversion.

**Two things in our tree used to stop us loading it, and W6a
([#1989](https://github.com/mudler/vllm.cpp/issues/1989)) has since discharged both**
(`e228d6893`, #2019). When this row first read the file, our GGUF reader had no
`case 20`, so the IQ4_NL that file uses for `ffn_down_exps` and for the n-gram table
failed at header parse; and `KeepQuantKDim` returned `-1` for `kEmbeddingTable`, so a
quantized gather table expanded to bf16 and 51.2B params became 102.4 GB. W6a added the
IQ4_NL and Q5_0 reader arms and a dequantizing gather, and made `kEmbeddingTable`
keep-quant eligible. **The file opens today on the CPU arm.** What is still owed is the
CUDA half: `EmbeddingKernelCuda` decodes no blocks, so `DeviceQuantGatherSupported` is
true for `kCPU` alone and the table keeps its expand-bf16 residency on CUDA. `## Owed`
carries that, and it is where this model's high-concurrency advantage lives.

It carries **no MTP weights** — zero `nextn`/`mtp` tensors of 1224 — while the
safetensors repo has 31. That is [#1993](https://github.com/mudler/vllm.cpp/issues/1993)'s
problem and `docs/USAGE.md` must say so beside the arm.

**The revision is now PINNED, and only a local digest is still owed.** The repo's
`lastModified` moved again after this row first read it, which is exactly the
re-quantized-in-place case AGENTS.md "Say which weights, and from where" names; a repo
id alone is not a pin. `## Owed` now carries revision
`8bdc666649440e9bdc97e16f3f75782c98478ff5` and the three per-shard sizes and digests.
Those digests were the Hub API's `lfs.oid` values and were not locally computed. That
debt is now DISCHARGED: all three were recomputed with `sha256sum` on the staged copy on
29 August 2026 and agree three for three, recorded in
[the ladder-arm evidence file](../../docs/bench-evidence/qwen4exp-llamacpp-ladder-arm-20260829.md). The `split.tensors.count
= 1224` above is on the same footing and is recorded there as UNVERIFIED, because shard
1 is the metadata shard and reports `n_tensors = 0`.

llama.cpp still has no *merged* `qwen4_exp` architecture -- two competing PRs are open
or withdrawn -- so authoring our own converter remains owed for arms nobody publishes.

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
5. **G4, speed against llama.cpp at its pin.** A denominator now exists and the
   earlier "there is no denominator" clause is superseded. **The oracle is
   [`llama-cpp-qwen4exp`](../oracles/llama-cpp-qwen4exp.md), pinned at
   `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3` (ggml-org/llama.cpp PR #27742), and NOT
   the stock [`llama-cpp`](../oracles/llama-cpp.md) pin this clause used to name.** That
   correction is measured, not stylistic: `llama-cpp` is pinned at released `b10451`,
   and a tree-wide grep for the string qwen4exp at that revision matches nothing (rc=1)
   while the same grep for qwen3vl matches three or more files (rc=0) as the control.
   Both were run against a fresh bare clone and are recorded in
   [the oracle file](../oracles/llama-cpp-qwen4exp.md) and in
   [#2060](https://github.com/mudler/vllm.cpp/issues/2060); they are not repeated as a
   command here, because a grep against a llama.cpp revision is not something this
   tree's gate runner can execute and a gate item that cannot fail gates nothing. The
   released oracle cannot name this architecture, so it cannot supply this denominator,
   and a gate that named it was naming a tool that refuses the model.

   `llama-cpp-qwen4exp` reads `gateable = yes` as of 29 August 2026, and both halves are
   recorded: it builds on CUDA for GB10 and `llama-server` at the pin loaded
   `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S and returned 64 coherent greedy tokens.
   [#2060](https://github.com/mudler/vllm.cpp/issues/2060) owed that run half and is
   discharged; the evidence is
   [`docs/bench-evidence/qwen4exp-llamacpp-ladder-arm-20260829.md`](../../docs/bench-evidence/qwen4exp-llamacpp-ladder-arm-20260829.md).
   **W6a has since made that file loadable on our side** (#2019), so the encoding
   precondition is met on the CPU arm; the gate itself still waits on G2, and no
   throughput, latency or memory number is admissible from this row until G2 passes.

   **`gateable = yes` is not a measurement.** That oracle's own record says so: one
   prompt, one repetition, five prompt tokens, one slot, no clock window and no
   contention control. Nothing in that evidence file may be quoted as a number.

   **The target is binding.** The developer's words, 2026-08-26, quoted rather than
   paraphrased because the wording is the requirement:

   > we should be faster than llama.cpp

   > especially at high concurrency

   Therefore:

   - **A concurrency LADDER is the headline, not a point.** c = 1, 4, 8, 16, 32 at
     minimum. A c=1 result neither confirms nor refutes this target. The harness walks
     the published online-serving grid, which is that set plus the c=2 the grid carries:
     `(1, 6) (2, 6) (4, 12) (8, 24) (16, 96) (32, 192)` as (concurrency, prompts).
   - **1,024 input tokens and 128 output tokens, three repetitions.** Stated here
     because this section used to be silent on the lengths while the runs used the
     published grid's values anyway, and an unstated input length is the axis that
     splits prefill from decode. These are `online_gate.INPUT_LEN`, `OUTPUT_LEN` and
     `REPETITIONS`, copied rather than chosen so a cell of the llama.cpp table can be
     read beside a cell of `docs/benchmarks/vllm-online-serving.md`.
   - **Prefill and decode reported separately**, because input length splits them and an
     aggregate hides which lever moved.
   - Memory is an axis: peak RSS and peak device bytes at each concurrency.
   - llama.cpp runs in its production configuration. A handicapped denominator is not a
     result, and this repository already has the `--enforce-eager` precedent for how that
     goes wrong.
   - Identical artifact, prompts, token counts, sampling and concurrency; idle host;
     reproduced with a same-binary A/B. "Identical prompts" means the same
     per-(concurrency, repetition) corpus partitions
     `online_gate.prepare_corpus_views` writes and refuses to let overlap — not one
     shared file replayed three times on one side against three disjoint thirds on the
     other — and the same `--num-warmups`, which `OnlineRun.num_warmups` sets to the
     concurrency.
   - **The denominator is TEXT-ONLY and this row is a multimodal port.** `/props` on the
     llama.cpp arm reports `"modalities":{"vision":false,"video":false,"audio":false}`.
     An arm that loads a vision tower does strictly more work per request, so the
     vllm.cpp cell paired against it must be a text-only configuration on the same
     UD-IQ1_S artifact. State which side ran what, or the ratio measures a configuration
     difference and reads as a performance one.

### Where the speed is expected to come from, and what would forfeit it

Four levers, from a source study of the two llama.cpp implementations (#27742 open,
#27739 closed by courtesy). **Both are UNMERGED**; each item is a reading of a pinned SHA
and not a measurement. Three of them constrained waves that had not started when this was
written; W3, W4 and W6a have since landed, and each lever below now records what its wave
actually did rather than what it was asked to do.

1. **Continuous batching and paged KV — the concurrency lever.** This engine mirrors
   vLLM's scheduler and block manager; llama.cpp's server allocates fixed parallel slots.
   That gap grows with concurrency rather than shrinking, which is where the target aims.
2. **The QSA consumer — the long-context lever, and the one this row could have
   forfeited by accident.** #27739 records that a sparse **mask** over a dense cache costs the same as
   dense attention under CUDA flash attention, because `flash_attn_mask_to_KV_max` only
   scans back to the first tile that is not all `-inf`. #27742 is mask-only and so buys
   correctness without decode speed. **W4 built the gather, not the mask** (#2030): the
   consumer counts at the key-row read, so the lever is preserved rather than forfeited.
3. **N-gram table residency.** #27742 makes the table CPU-resident by tensor class
   regardless of `-ngl`, so every token's 16 gathers are host work. At IQ4_NL the table is
   ~28.8 GB inside a 67.56 GiB file against ~119.6 GiB usable, so it can be
   device-resident and quantized. At batch B that is 16xB uncoalesced random gathers, so
   host-versus-device here is a scaling difference, not a constant. **W6a asserted that
   decision rather than defaulting it** (#2019): the table stays quantized and is gathered
   in place on CPU. The CUDA arm is unbuilt, so on CUDA the table still expands, and this
   lever is only half collected.
4. **The hyper-connection write-back.** Both PRs materialise the rank-1 update as a
   `repeat_4d` + `mul`: 96 materialised `[2560, 4, T]` broadcasts per forward at 48
   layers x 2 sites. **W3 landed leaving the fused seam reachable rather than built**
   (#2045): `GatedResidualWriteBackInPlace` is the primitive, and no device kernel
   replaces it yet, which `## Owed` carries.

**No ceiling may be declared** if a first measurement disappoints. An apparent
same-artifact limit is an unresolved implementation difference with a next traceable
hypothesis, every time.

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

## Mutation record — W5b-1 (#2110)

The cross-TU GDN seam (`RunGdnBlockPaged` / `BuildGdnStepInputs`). Every row
below was measured by the W5b-1 REPAIR on the merged head, not relayed from the
implementer or from the fresh review, because the review found that the case's
own comment named an outcome no mutation of that function can produce.

Method, unchanged from the W5a section above: one textual change applied to a
pristine `src/vllm/model_executor/models/qwen3_5.cpp`, rebuilt with the BUILD RC
read before any test result, run, restored from a byte-identical copy,
`sha256sum -c`-verified at
`d0db911160f326ab83e3fdc13ee8dd4df0f25a9f292790eb1ccf3a08a087cdfc`, rebuilt and
re-run green. The un-mutated baseline on this head is
`test_qwen3_5_gdn_spec_routing` 7 cases / 82 assertions,
`test_qwen27_paged_forward` 31 / 770, `test_qwen35_moe_gdn_ba_owner` 1 / 23,
`test_qwen3_5_decode_graph_seam` 10 / 156.

| # | mutation | target | result |
|---|---|---|---|
| A | perturb the gated-RMSNorm epsilon inside `GdnBlockPaged` (`+ 1e-3F` at its one definition) | `test_qwen3_5_gdn_spec_routing`, `test_qwen27_paged_forward` | **RED on the EXISTING cases: 1 of 7 (the MIXED spec+prefill case, 2 assertions) and 5 of 31 (6 assertions).** The Qwen3.5/3.6 forward still runs this block, so the block the wrapper exposes is the live one. **The NEW seam case stays GREEN, and it must:** both arms of its comparison enter `GdnBlockPaged`, so no uniform mutation of that function can separate them. The implementer's comment claimed this mutation reds the new case together with the spec-routing cases; that claim is false and is corrected in the test |
| B | make the wrapper stop delegating — allocate a `[T,H]` output at `GdnOutDType()` and `Zero()` it instead of calling `GdnBlockPaged` | `test_qwen3_5_gdn_spec_routing` | **RED on exactly the new case, 10 assertions**, the 6 pre-existing cases green. Six are the output/SSM/conv bit-for-bit comparisons at both gate dims; four are the `dh_fp8` sub-case, which now sees the wrapper RETURN instead of refusing. This is the mutation that reds the new case |
| C | make the wrapper stop FORWARDING `dh_fp8` (`GdnBlockPaged(..., T, nullptr)`) | `test_qwen3_5_gdn_spec_routing` | **RED on exactly the new case, 2 of 82 assertions.** Before the repair this mutation SURVIVED at 7 / 74: the comparison case runs a BF16 weight set where `ProjectGdnQkvz` never reads `h_fp8`, so the only production argument the wrapper forwards that nothing gated was the fp8 one. The sub-case added by this repair closes it |

**Why `dh_fp8` is gated by a refusal and not by a number.** `dh_fp8` selects
between two mutually exclusive production leaves inside `ProjectGdnQkvz`:
non-null takes `MatmulFp8CutlassPreQuantD`, null takes `MatmulFp8CutlassD`. Both
open with a `VT_CHECK` on `vt::OpRegistered(kMatmulFp8CublasLt, device)`, and
that op is registered for `kCUDA` alone
(`include/vllm/model_executor/models/dense_fp8_gemm.h`), so on a host queue the
fp8 GDN tower computes no value the case could compare. It does produce a
refusal that names the leaf that refused, and the two leaves name themselves
differently. BOTH directions are asserted, so the observable discriminates
rather than merely observes. A CUDA host can strengthen this to a numeric
comparison; that is owed, not done here.

**Mutation B's build is the trap this record exists to name.** Removing the
delegation leaves `gdn_meta` and `state` unused, and this tree builds with
`-Werror=unused-parameter`, so the first attempt does not compile — and a
mutation that never built leaves the STALE binary printing green. Read the build
RC before any test result. The fresh review hit exactly this and read a false
7 / 74 pass.

## Mutation record — W5b-3 (#2156)

The PLE dilated depthwise conv as `vt::Qwen4ExpPleConv`. Sixteen mutations, one
at a time, each proved APPLIED by a sha256 that moved, each build's exit status
read BEFORE any test result, and the tree restored byte-for-byte and re-verified
by sha256 after every one. Re-measured on the final head. Suites:
`test_qwen4_exp_ple_device` (the new device gate, 10 cases / 538 assertions
green) and `test_qwen4_exp_ple` (the W2 host suite, 9 / 395 green), the second
present as a control that no mutation of the device arm can move.

**Five of the sixteen failed to BUILD on the first pass, and that is a result
about the harness rather than about the code.** `-Werror` turns "the mutation
made a variable unused" into a link that never happens, the runner then executes
the STALE binary, and a stale binary prints green. M1, M2, M5, M7 and M13 each
did exactly that. They are re-run with the one `(void)x;` or `[[maybe_unused]]`
that silences the warning and changes nothing the mutation is about, and only
the second reading is recorded. This is the third time in this campaign that a
build failure has presented as a pass; reading the build rc first is what caught
it.

| # | mutation | file | verdict |
|---|---|---|---|
| M1 | `hist[t + k * dilation]` → `hist[t + k]`: the taps read at unit stride | kernel | **RED, 5 of 10 cases, 7 of 544 assertions** |
| M2 | `dilation = args.dilation` → `dilation = 1`: the arg is never read | kernel | **RED, 5 of 10 cases, 7 of 544** |
| M3 | the tap order reversed, `weight[c*K + k]` → `weight[c*K + (K-1-k)]` | kernel | **RED, 5 of 10 cases, 9 of 546** |
| M4 | the state write-back one column early, `hist[tokens+j]` → `hist[tokens+j-1]` | kernel | **RED, 4 of 10 cases, 263 of 538** |
| M5 | the silu dropped from the store | kernel | **RED, 5 of 10 cases, 9 of 546** |
| M6 | the state keeps the ACTIVATED value instead of the raw conv input | kernel | **RED, 4 of 10 cases, 263 of 538** |
| M7 | `conv_state_indices` ignored: row `s` for sequence `s` unconditionally | kernel | **RED, 1 of 10 cases, 3 of 538** |
| M8 | the per-sequence token offset dropped on the `x` load | kernel | **RED, 1 of 10 cases, 113 of 538** |
| M9 | the tap accumulator narrowed from `double` to `float` | kernel | **RED, 1 of 10 cases, 1 of 538** — the model-width case, which asserts BIT-IDENTITY with the host reference by `memcmp`. No golden comparison at C = 16 can see this; the 10240-channel agreement check is the only thing that does |
| M10 | the empty-segment early-out removed | kernel | **SURVIVED — and it is an EQUIVALENT MUTANT, not a gate hole.** With `tokens == 0` the span is `state_len`, the window loop does not execute, and the write-back reads `hist[0 + j]`, which is the column it then writes: the two programs compute the same function. The dispatcher refuses a decreasing `query_start_loc`, so `0` is the only value that reaches the branch. It is kept as a PERFORMANCE early-out — at 10240 channels a padded batch row would otherwise cost 184k pointless float copies per layer — and the kernel comment says that in those words, because the comment that stood there first claimed it stopped the cache being shifted and it does not. The repair is M16, which mutates the same territory in a way a test can see |
| M11 | the `(K-1)*dilation` state-width check widened to `>= K-1` | dispatcher | **RED, 1 of 10 cases, 1 of 538** — the Mamba-shaped-state refusal |
| M12 | the `query_start_loc` bounds check removed | dispatcher | **RED, 1 of 10 cases, 1 of 538** |
| M13 | the `conv_state_indices` range check removed | dispatcher | **RED, rc = 134 (SIGABRT), 1 of 10 cases, 1 of 535** — the refusal assertion reports `did NOT throw at all!`, and the unchecked row index (7 into a cache of 3) then writes past the allocation, which glibc catches as `double free or corruption (out)` and turns into `SIGABRT`; doctest prints `FATAL ERROR: test case CRASHED: SIGABRT`. **The SIGNAL is not stable and the row must not be read as if it were.** The first record here said `rc = -6`, which was a negative `WTERMSIG` written where a shell exit status belongs; the fresh reviewer of this wave measured `rc = 139` (`SIGSEGV`, core dumped) on the same case and the same 1-of-10 / 1-of-535 counts; this re-run measured 134. All three are the same defect. An out-of-range row index writes at `row_stride * 7` past a three-row cache, and whether that lands in unmapped memory (`SIGSEGV`) or in allocator bookkeeping the next free checks (`SIGABRT`) is a property of the heap layout, not of the mutation. What is stable, and what the row is actually evidence for, is the assertion count: the refusal is the ONLY thing standing between a caller error and undefined behaviour, which is why it is a check rather than a comment. The re-run deleted the whole `if (conv_state_indices != nullptr)` block, declaration included, so unlike the first pass it needed no `(void)` silencer and built at rc 0 — the build rc was read before the run rc, because a stale binary prints green |
| M15 | the segment loop stops after the first sequence | kernel | **RED, 2 of 10 cases, 114 of 538** |
| M16 | an empty segment RESETS its cache row instead of leaving it | kernel | **RED, 1 of 10 cases, 1 of 538** — M10's repair: the plausible defect in that territory is clobbering a padded row, and the empty-segment case sees it |
| M14 | **REACHABILITY**: the `RegisterOp(OpId::kQwen4ExpPleConv, DeviceType::kCPU, ...)` line deleted | kernel | **RED, 9 of 10 cases, only 9 assertions reached** — every case that calls the op throws `vt: no kernel for op Qwen4ExpPleConv (id 134) on device cpu`. `[[maybe_unused]]` on the kernel is required or `-Werror=unused-function` fails the build and the stale binary reads green |

The M14 shape is the load-bearing reachability proof AVAILABLE AT THIS LAYER,
and it is not the one AGENTS.md `## Nothing lands dead` really wants. Deleting a
production call site is impossible here because there is no production call site
— see `## Owed` — so what M14 proves is that the tests enter the op through the
dispatcher and the registry rather than through the kernel function, which is
the strongest statement this slice can make.

**The RED that came first.** Before the kernel existed, with the OpId, the args
struct, the dispatcher and the test all present, `test_qwen4_exp_ple_device`
reported 8 of 9 cases failing with
`vt: no kernel for op Qwen4ExpPleConv (id 134) on device cpu (type 0)` at
`src/vt/op_provider.cpp:577`. The one case that passed was the refusals case,
whose subcases all throw in the dispatcher before reaching `GetOp` — which is
itself the evidence that the geometry checks are in the dispatcher and not in
the kernel.

## Mutation record — W5b-2 (#2123)

The device arm of the gated-residual stream, `vt::Qwen4ExpGatedResidual` and
`vt::Qwen4ExpGatedResidualWriteBack`. Method as in the two sections above: one
textual change applied to a pristine tree, proved applied by a **sha256**
comparison rather than by `git diff` — the kernel translation unit is NEW on
this branch and an untracked file has an empty diff no matter what is written
into it — then `touch`ed, rebuilt, run, restored from a byte-identical copy,
`sha256sum`-verified, `touch`ed again and rebuilt. Without that second touch
ninja skips the rebuild and the mutations ACCUMULATE, which fails toward RED and
makes a weak gate read strong. Every row was re-measured on the FINAL head, not
at the point in the repair where its fix landed.

**Two mutations survived the first battery, and both are repaired here rather
than recorded and left.** The table below is the second battery; the first
battery's verdict is in the right-hand column where it differs, because a
mutation table whose only rows are reds is a table nobody re-ran.

Target `src/vt/cpu/cpu_qwen4_exp.cpp` unless stated; suite
`tests/vllm/models/test_qwen4_exp_hc_device.cpp`, 9 cases / 87 assertions green.

| # | mutation | result | first battery |
|---|---|---|---|
| M1 | `/ hc_count` moved OUTSIDE the SiLU | RED, 2 of 9 cases, 4 assertions | RED |
| M2 | a `/ hc_count` ADDED to the up-projection sigmoid | RED, 2 of 9, 4 | RED |
| M3 | the branch collapse made a SUM instead of a MEAN | RED, 4 of 9, 7 | RED |
| M4 | the gate multiplied against the RAW stream, not the normed one | RED, 4 of 9, 7 | RED |
| M5 | injection sigmoid loses its `/ hc_count` | RED, 1 of 9, 3 | RED |
| M6 | injection loses its `2 *` | RED, 2 of 9, 5 | RED |
| M7 | `eps` moved OUTSIDE the rsqrt (added to the norm, not the mean square) | **RED, 1 of 9, 2** | **SURVIVED** |
| M8 | grouped norm collapsed to a whole-row norm over `hc*H` | RED, 4 of 9, 10 | RED |
| M9 | the per-group sum of squares accumulated in `float` | **RED, 1 of 9, 1** | **SURVIVED** |
| M10 | write-back reads `block_out` backwards | RED, 2 of 9, 5 | RED |
| M11 | write-back assigns instead of accumulating | RED, 2 of 9, 5 | RED |
| M12 | every token reads token 0's stream (the batch axis dropped) | RED, 4 of 9, 12 | RED |
| M13 | the normed stream rounded through the stream dtype (upstream's `.type_as(x)`) | RED, 1 of 9, 8 | RED |
| M14 | `x / hc` replaced by `x * (1.0f / hc)` | **SURVIVED**, 9/9, 87 | SURVIVED |
| M15 | the write-back op's `RegisterOp` line deleted, **and `[[maybe_unused]]` added to `Qwen4ExpGatedResidualWriteBackKernel`** | RED, 2 of 9, and the suite runs 74 assertions instead of 87 | RED |
| M16 | the SAME eps mutation in the HOST reference `qwen4_exp_hc.cpp`, against `test_qwen4_exp_hc` | **RED, 3 of 15 cases, 10 assertions** (pre-repair, at `origin/main`: RED, 2 of 14, 2 — W3's `big_eps` probe already gated it) | n/a |
| M17 | M16's mutation against the DEVICE suite | SURVIVED, 9/9, 87 | n/a |
| M18 | the host reference's mean collapse made a SUM, against the DEVICE suite | RED, 1 of 9, 2 | n/a |

**What the two repairs were, and why the first battery could not see either.**

*M7, the epsilon placement.* `eps` goes inside the rsqrt, added to the MEAN
SQUARE (`torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)`,
`modeling_qwen4_exp.py:170`); the plausible slip adds it to the norm instead.
Golden cases A, B and C all draw the stream at `hyper_scale = 1.7`, where the
mean square is O(1) and `eps = 1e-6` moves the answer by about 5e-7 — a third of
the suite's own `kTol`, so the wrong spelling passes. The same is true of every
GOLDEN case in the W3 host suite, and its `Variant` sweep has no flag for it
either — but **W3 did not ship only goldens.** It shipped a deliberate large-eps
probe (`test_qwen4_exp_hc.cpp:268-276`) that pins the placement against
`NormRefD` at `big_eps = 4.0`, and on a reconstructed pre-repair tree the eps
mutation reds it: `CHECK( 0.802185 < 1e-05 )` at :276, **2 of 14 cases and 2
assertions** over the whole suite. The hole was the DEVICE arm's, which had no
such probe. M16 is in this table because case D raises the host arm's answer from
those 2 assertions to 10, not because the host arm was silent. The repair is a
fourth golden case, `D`, drawn at `hyper_scale = 0.01` — mean square ~1e-4, so
`eps` is 1% of it and the
two spellings separate by ~0.5%, three orders over `kTol`. `scripts/gen-qwen4-exp-hc-goldens.py`
grew one `hyper_scale` parameter, defaulted to the existing 1.7, and A/B/C
regenerate byte-identically (verified: the only diff in those three blocks is the
comment line that now prints the scale). Case D is driven by BOTH suites.

*M9, the accumulator width.* The per-group sum of squares runs over 2560 terms in
the real model and `double ss` -> `float ss` is a one-word edit that changes
nothing measurable on ordinary data. The W3 suite already gates it — at the real
group size, on magnitude-separated data — but it gates the HOST reference, not
this kernel, and the device suite's own real-width case draws uniform inputs where
the two accumulators agree. The repair is a device case built on the same
principle and with one addition: `mix_down` and `mix_up` are ZEROED, so `silu(0)`
is exactly 0, every gate is `sigmoid(0) = 0.5` exactly, and `mixed` is a four-term
f32 mean of the normed stream with the reduction under test as its ONLY remaining
error source. Without that the two f32 projections over `flat = 10240` contribute
their own ~1e-6 and sit on top of the signal. Measured on exactly that data:

| | value |
|---|---|
| max abs of the double reference | 3.2895e+01 |
| ours, `double` accumulator | 1.173e-06 (3.6e-08 relative, ~0.6 ulp) |
| the same kernel with `float ss` | 6.702e-04 (**571x worse**) |

The bound is 1e-5: 8.5x above the first and 67x below the second.

**M15's recipe needs the attribute, and without it the row is unmeasurable.**
Deleting the `RegisterOp` line alone leaves the kernel defined and uncalled, and
this tree builds with `-Werror=unused-function`, so the translation unit does not
compile: `error: 'Qwen4ExpGatedResidualWriteBackKernel' defined but not used`.
The build then exits non-zero, the STALE device binary is still on disk, and
running it prints 9 / 87 SUCCESS — the false green the W5b-1 section above names
in the same words. Read the build RC first, and add `[[maybe_unused]]` to the
kernel's declaration so the deletion under test is the registration and nothing
else. With the attribute the recorded result reproduces exactly: build RC 0,
2 of 9 cases red on `no kernel for op Qwen4ExpGatedResidualWriteBack`, 74
assertions instead of 87.

**M14 is a real survivor and it stays one.** `x / hc` and `x * (1.0f / hc)` differ
by at most one ulp for an `hc` that is not a power of two, which is golden case B
at `hc_count = 3`; the difference propagates through a SiLU and a projection and
arrives at `mixed` some four orders below `kTol`. No comparison against f32
goldens can separate them, and tightening `kTol` far enough would fail the
unmutated kernel first (the measured unmutated max abs difference over case B is
2.384e-07). The correct spelling is kept because upstream spells it `/
self.hc_count` and the host reference already argues the point, not because a gate
proves it. Recorded so nobody reads its absence from the red list as coverage.

**M17 is the honest shape of an agreement check.** It applies M16's epsilon
mutation to the host reference and runs the DEVICE suite, which reaches that
reference only through its two model-width cases — where `eps = 1e-6` against a
mean square of ~0.33 is invisible, exactly as it is in cases A/B/C. It says the
model-width comparison cannot see an epsilon defect, and M18 is the row that says
the comparison is nonetheless LIVE: a sum-for-mean change in the same reference
reds it. Both rows are here because a single survivor with no companion reads as
an instrument that is not wired up.

## Mutation record — W5b-4 (#2167)

Qwen Sparse Attention on the device arm: `vt::Qwen4ExpQsaCompress`,
`vt::Qwen4ExpQsaGatherAttention`, and the indexer COMPOSED from
`vt::DsaIndexerLogits` + `vt::DsaTopkSelect` (see `### W5b-4 correction` above
for why those two are not re-implemented here). Method as in the sections above:
one textual change applied to a pristine tree, proved applied by a **sha256**
comparison rather than by `git diff` — both new files are UNTRACKED on this
branch and an untracked file has an empty diff no matter what is written into it
— then rebuilt, run, restored from a byte-identical copy and `sha256sum`-verified
against the pre-mutation digest. **The build return code is read BEFORE any test
result**: the first battery had three mutations that failed to compile under
`-Werror` (`half`, `groups` and `prev` become unused when the line that reads
them is replaced), and a `ninja` failure leaves the previous binary in place, so
each of the three would otherwise have run a STALE binary and reported a pass.

Every row below is re-measured on the FINAL head, after the repair, not at the
point in the wave where its fix landed. Target `src/vt/cpu/cpu_qwen4_exp_qsa.cpp`
unless stated; suite `tests/vllm/models/test_qwen4_exp_qsa_device.cpp`,
**12 cases / 4697 assertions green** after the fresh review's repair added the
unmapped-tail probe. The per-mutation figures in the table below were measured on
the 11-case suite, before that case existed; the three rows the repair added or
re-measured (the probe row, M11 and M11c) name the 12-case suite explicitly.

| # | mutation | build rc | result | first battery |
|---|---|---|---|---|
| M1 | pool stores a SUM, not a mean (drop the / compress_ratio) | 0 | RED, 2 of 11 cases, 9 assertions | RED |
| M2 | eps OUTSIDE the rsqrt instead of added to the mean square | 0 | RED, 2 of 11 cases, 9 assertions | RED |
| M3 | RoPE at the block's LAST position instead of its first | 0 | RED, 4 of 11 cases, 53 assertions | RED |
| M4 | vLLM norm polarity `* w` instead of upstream's `* (1 + w)` | 0 | RED, 6 of 11 cases, 1516 assertions | RED |
| M5 | rotate_half loses its minus sign | 0 | RED, 3 of 11 cases, 39 assertions | RED |
| M6 | GPT-J adjacent-pair rotation (DeepSeek-V4's) instead of NeoX half-split | 0 | RED, 3 of 11 cases, 35 assertions | build failed |
| M7 | the pool's `.to(dtype)` bf16 round-trip dropped | 0 | RED, 1 of 11 cases, 2 assertions | RED |
| M8 | OVERLAPPING pooling window (DeepSeek-V4's shape), stride 1 not CR | 0 | RED, 5 of 11 cases, 1548 assertions | RED |
| M9 | the RMS norm's `.type_as(x)` bf16 rounding dropped | 0 | RED, 1 of 11 cases, 2 assertions | RED |
| M10 | the NoPE dims are zeroed instead of carried through untouched | 0 | RED, 5 of 11 cases, 1479 assertions | RED |
| M11 | THE LOAD-BEARING ONE: a dense masked walk reporting the SPARSE read count | 0 | RED, 1 of 11 cases, 256 assertions; **re-measured on the 12-case suite: RED, 2 of 12 cases, 257 assertions** — the NaN case and the unmapped-tail probe | **SURVIVED** |
| M11b | the same dense masked walk, with the counter left AT the read site | 0 | RED, 3 of 11 cases, 260 assertions | n/a |
| M11c | THE FETCH, NOT THE MULTIPLY: prefetch every cached row and discard it, then gather honestly | 0 | RED, **1 of 12** cases — the unmapped-tail probe ALONE; the NaN case and every read-count case pass | n/a |
| M12 | the always-attended ragged tail dropped | 0 | RED, 5 of 11 cases, 899 assertions | RED |
| M13 | block b expands to tokens [b, b + CR) instead of [CR*b, CR*b + CR) | 0 | RED, 4 of 11 cases, 1538 assertions | RED |
| M14 | block b expands to CR - 1 tokens (the off-by-one) | 0 | RED, 5 of 11 cases, 1926 assertions | RED |
| M15 | GQA head mapping `h % HKV` instead of `h / groups` | 0 | RED, 4 of 11 cases, 1154 assertions | build failed |
| M16 | the softmax scale dropped from the logit | 0 | RED, 3 of 11 cases, 1907 assertions | RED |
| M17 | reads counted once per row instead of once per pass | 0 | RED, 2 of 11 cases, 4 assertions | RED |
| M18 | the gather visits its rows in DESCENDING order | 0 | RED, 3 of 11 cases, 1577 assertions | RED |
| M19 | the ASCENDING/in-range block refusal deleted | 0 | RED, 1 of 11 cases, 1 assertions | build failed |
| M20 | the COMPLETE-blocks refusal deleted from the compressor dispatcher (`src/vt/ops.cpp`) | 0 | RED, 1 of 11 cases, 1 assertions | RED |
| M21 | the rotary-fits-the-index-head refusal deleted (`src/vt/ops.cpp`) | 0 | RED, 1 of 11 cases, 1 assertions | RED |
| M22 | the explicit-scale refusal deleted from the gather dispatcher (`src/vt/ops.cpp`) | 0 | RED, 1 of 11 cases, 1 assertions | not applied |
| M23 | the cos/sin coverage refusal deleted (`src/vt/ops.cpp`) | 0 | RED, 1 of 11 cases, 1 assertions | RED |
| M24 | REACHABILITY: the compressor's RegisterOp deleted | 0 | RED, 10 of 11 cases, 0 assertions | RED |
| M25 | REACHABILITY: the gather's RegisterOp deleted | 0 | RED, 6 of 11 cases, 1 assertions | RED |
| M26 | CONTROL: inherit DeepSeek-V4's n_head_scale, which QSA has no tensor for (the TEST) | 0 | **SURVIVED**, 11/11, 4427 | **SURVIVED** |
| M27 | the indexer's `weights` stop being ones, so the fold no longer collapses (the TEST) | 0 | RED, 3 of 11 cases, 1269 assertions | RED |
| M28 | the scoring window becomes the whole cache, not the visible complete blocks (the TEST) | 0 | RED, 4 of 11 cases, 0 assertions | RED |

**THE SURVIVOR, AND ITS REPAIR.** M11 is the defect this whole wave exists to
prevent, and in the first battery it SURVIVED: 10 of 10 cases, 4167 of 4167
assertions, against a body that walked every one of the `kv_len` cached rows with
a `-inf` mask and reported `sel.size() * 2` per head as its read count. That is
W4's M22c reproduced one layer up, and the reason is structural rather than
careless. A `keys_visited` the kernel writes cannot convict the kernel that
writes it, whatever the counter's placement in the SHIPPED code, because a
mask-shaped port changes the loop and the counter together — that is what a
mask-shaped port IS. And no value comparison can convict it either: `exp(-inf -
m)` is exactly +0 and adding an exact zero changes no accumulator, so a mask
agrees with a gather value for value, which is precisely why a token gate lets
one through.

The repair is an observable of the WALK rather than of the bookkeeping. The case
`vt::Qwen4ExpQsaGatherAttention: the unselected rows are NaN and the answer is
finite` poisons every cached row the selection does not name with `NaN`, in both
K and V, for a single query token whose complement is therefore well defined
(23 cached, 11 attended, 12 poisoned). A gather never addresses those rows and is
bit-identical to the same gather over a clean cache. A mask reads every value row
and accumulates `w * v` with `w == 0.0f`, and `0.0f * NaN` is `NaN` in IEEE-754,
so its output is `NaN` in every lane. M11 reds on it.

M11b is the companion that says the counter is nonetheless live: the SAME dense
masked walk with `++reads` left at the read site reds the read-count cases
directly, so the shipped counter is a function of the walk and not a restatement
of the selection. A single survivor with no companion reads as an instrument
nobody wired up, which is why both rows are here.

**THE LIMIT OF THE NaN PROBE, AND THE PROBE THAT CLOSES IT.** The NaN poison
proves a row was not multiplied into the accumulator. It does not prove the row's
bytes were never fetched: a body that loads every row and discards the unselected
ones before the multiply passes it, and the loop counter is not the cost
llama.cpp #27739 measures anyway — the key-row traffic is.

The wave's first draft recorded that gap as blocked, on the reasoning that the
structural version is a PAGED cache whose unselected blocks are not mapped, and
that the block-table store is owed and waits on
[#2131](https://github.com/mudler/vllm.cpp/issues/2131). **That is true of the
PRODUCTION cache and false of a test instrument**, and the fresh review proved it
by building one. `vt::Qwen4ExpQsaGatherAttention: the gather never FETCHES an
unmapped unselected row` `mmap`s its own page-aligned K and V caches at kv_len
3000 — a multiple of `compress_ratio`, so the always-attended ragged tail is
empty — selects blocks `0..511`, and `mprotect(PROT_NONE)`s the 59 whole pages
(241664 bytes, `[524288, 768000)`) that lie strictly inside the unselected run
`[2048, 3000)`, in BOTH caches. The shipped kernel walks past the hole:
`keys_visited` 16384 against a dense 24000, and its output is bit-identical to
the same call over the unguarded mapping. M11's dense masked walk dereferences
the first guarded row and takes SIGSEGV.

The construction is forced by the kernel, not chosen: the gather addresses the
cache as `(p * HKV + kvh) * DH + d` and never reads `key.stride[0]`, so a guard
page BETWEEN rows is unavailable and the unselected rows have to form one
contiguous tail.

**M11c is what says the probe is not a restatement of the NaN case.** It
prefetches every cached row into a discarded accumulator and then gathers
honestly with the honest counter — the exact body the paragraph above names as
the NaN probe's blind spot. It reds **one** case out of twelve, the unmapped-tail
probe, and passes the NaN case, every read-count case and every value case. The
two probes are therefore ordered, not redundant: NaN convicts the multiply, the
unmapped tail convicts the fetch.

**A fault has to be a failing assertion, not a dead binary.** doctest installs a
fatal-condition handler around every case, and left in place it turns the
mutant's SIGSEGV into `FATAL ERROR: test case CRASHED` and abandons the rest of
the run — every remaining case reads as skipped, so one convicted mutant costs
the verdict on every other property in the file. This was measured, not assumed:
the first build of the probe omitted the handler and exited 139 with eleven cases
unreported. The probe installs its own `SIGSEGV`/`SIGBUS` handler around the one
call, `siglongjmp`s back into the case, restores doctest's handlers, and reports
`CHECK_FALSE(faulted)`. On a platform without POSIX `mmap`/`mprotect` the case is
declared `doctest::skip()` rather than compiled out, because a probe that cannot
run must say so.

**THE OTHER SURVIVOR IS A DELIBERATE CONTROL.** M26 inherits DeepSeek-V4's
`n_head_scale = n_head ** -0.5` into the composed indexer, which QSA has no
tensor for, and the suite stays green. That is not a hole: top-k is invariant
under a positive rescale of every score, so the mutation cannot move a selection
by construction — the same blindness W4 already recorded for `QsaBlockScores`,
and the reason that constant is gated by a hand-derived VALUE case in the host
suite rather than by any selection. M27 is its paired red: making the indexer's
`weights` non-uniform breaks the fold's collapse to a single constant and reds
3 of 11 cases, so the composition's `weights == 1` is load-bearing rather than
decorative.

**WHAT THE CONTEXT LENGTH BUYS, measured.** M12 (the ragged tail dropped) and
M14 (a block expanded to `CR - 1` tokens) both red, and both would red at the
golden shapes alone. The case that only the released-config context can carry is
the sparsity itself: at kv_len 2051 — the sub-budget control — `keys_visited`
equals the dense figure exactly, so every read-count assertion in this file is
trivially true there and a mask passes them all. At kv_len 3002 the gather reads
2050 of 3002 rows per query token and the same assertions bite. A QSA gate that
never crosses 2048 is not a weaker gate; it is not a gate.

## Mutation record — W5b-5 (#2211)

`Qwen4ExpTextAttention` as one production block, and the first place under
`src/` that COMPOSES the QSA indexer. Method as in the sections above: one
textual change applied to a pristine tree, proved applied by a **sha256 that
moved** (the file is NEW on this branch and an untracked file has an empty diff
whatever is written into it), the file `touch`ed so ninja cannot skip the
rebuild, the **BUILD RETURN CODE READ BEFORE ANY TEST RESULT**, then restored
from a byte-identical copy and `sha256sum`-verified against the pre-mutation
digest `ab132cafcd327dda…`. Every row was re-measured on the FINAL head, after
the repair. Target `src/vllm/model_executor/models/qwen4_exp_qsa_block.cpp`;
suite `tests/vllm/models/test_qwen4_exp_qsa_block.cpp`, **8 cases / 2828
assertions green** at the time this table was measured; the fresh-review repair
recorded below took that to **8 / 2831**.

**THE RED CAME FIRST, AND IT WAS MEASURED RATHER THAN ASSERTED.** With the suite
and the header in place and both function BODIES replaced by a refusal, build
rc 0, the run reported **7 of 7 cases failing, 8 of 11 assertions**, every one on
`qwen4_exp qsa block: not composed yet`. Nothing in this file has ever passed
against an absent composition.

**Two mutations failed to BUILD on the first pass, which is the fourth time in
this campaign that a build failure has presented as a pass.** M3 leaves `one`
unused and M4 leaves `kl` unused; this tree builds with `-Werror`, ninja then
leaves the STALE binary on disk, and a stale binary prints green. Both are re-run
with the one `(void)x;` or `[[maybe_unused]]` that silences the warning and
changes nothing the mutation is about, and only the second reading is recorded.
Reading the build rc before the run rc is what caught it.

| # | mutation | build rc | result |
|---|---|---|---|
| M1 | **setting 2**: inherit DeepSeek-V4's `n_head_scale = n_head ** -0.5`, which QSA has no tensor for | 0 | **RED, 1 of 8 cases, 2 assertions** — the LOGITS VALUE case alone, which is the whole point of it. This is the repair for M26, which survived at the op layer because no selection can move under a positive rescale |
| M2 | **setting 3**: the softmax scale dropped from the fold | 0 | **RED, 1 of 8, 2** — again the value case alone |
| M3 | **setting 1**: the indexer `weights` stop being ones, so the per-head fold no longer collapses to one constant | 0 | RED, 3 of 8, 7 |
| M4 | **setting 4**: the scoring window becomes the WHOLE cache instead of the visible complete blocks | 0 | RED, 5 of 8 — by REFUSAL, not by assertion: a query then selects a block it cannot see and the gather's own in-range check throws. 113 assertions reached |
| M5 | setting 4 at the other end: `win_start` skips the first block | 0 | RED, 6 of 8, 2 assertions of 113 — the value case reds, then the gather refuses a query that attends nothing |
| M6 | the pooled key's rope span derived from the INDEXER head width instead of the model's | 0 | RED, 8 of 8, 11 assertions reached — a REFUSAL mutation: it trips the compressor's cos/sin coverage check |
| M6b | the pooled key is never roped at all (`rotary_dim = 0`), which is IN RANGE | 0 | RED, 8 of 8 — the arithmetic companion to M6, so the territory is not gated by a refusal alone |
| M7 | the pooled key's bf16 round-trip dropped | 0 | RED, 1 of 8, 1 |
| M8 | the ragged tail rounded UP into a block that does not exist yet | 0 | RED, 8 of 8, 11 reached — a REFUSAL mutation: this file's own `RowsView` range check |
| M8b | the LAST complete block dropped, which is IN RANGE | 0 | RED, 5 of 8, 4 assertions — the arithmetic companion to M8 |
| M9 | the indexer q norm applies `out * w` instead of upstream's `out * (1 + w)` | 0 | RED, 1 of 8, 1 |
| M10 | the model q norm loses the `+1` fold | 0 | RED, 2 of 8, 3 |
| M11 | the model k norm loses the `+1` fold | 0 | RED, 2 of 8, 3 |
| M12 | the side cache stores a NORMED indexer key, which the compressor then norms again | 0 | RED, 4 of 8, 38 |
| M13 | the indexer key ignores `past_len` and overwrites row 0 | 0 | **RED, 1 of 8, 2 — after the repair. It SURVIVED the first battery**, see below |
| M14 | the attention softmax scale taken from the INDEXER head dim | 0 | RED, 2 of 8, 3 |
| M15 | the query and the OUTPUT GATE halves of `q_proj` swapped | 0 | RED, 2 of 8, 3 |
| M16 | the sigmoid OUTPUT GATE dropped entirely | 0 | RED, 2 of 8, 3 |
| M17 | the key is never roped, only the query | 0 | RED, 2 of 8, 3 |
| M18 | GPT-J adjacent-pair rotation (DeepSeek-V4's) instead of NeoX half-split | 0 | RED, 3 of 8, 131 |
| M19 | the value cache ignores `past_len` | 0 | RED, 1 of 8, 1 |
| M20 | **THE LOAD-BEARING ONE**: the consumer is handed EVERY VISIBLE block — a dense walk wearing a gather's clothes | 0 | **RED, 3 of 8, 130 assertions** — the golden case at over-budget, the NaN-poison case and the released-config sparsity case |
| M21 | the key is roped into a SCRATCH, so the cache keeps the un-roped key | 0 | RED, 2 of 8, 3 |
| M22 | **the rope cross-check's production call site deleted** — the fresh-review repair below | 0, after `[[maybe_unused]]` silences the `-Werror=unused-function` this leaves | **RED, 1 of 8, 2** — both new refusal subcases stop throwing |

**THE SURVIVOR, AND ITS REPAIR.** M13 writes the indexer key at cache row 0
instead of row `past_len`, and in the first battery it SURVIVED: 8 of 8 cases,
2805 of 2805 assertions. The reason is structural rather than careless. Every
golden case is a PREFILL, where `past_len` is 0 and the two spellings are the
same expression; the one case with `past_len > 0` compared the block OUTPUT at a
bf16-sized relative bound, and one wrong pooled key moves the selection by one
block out of five and leaves the answer inside it. The repair is an observable of
WHERE the write landed rather than of what it was worth: the decode case now
compares the side cache ROW FOR ROW against `k...IdxKRaw`, the oracle's own raw
indexer keys, after a split prefill/decode. M13 reds on it.

**WHAT THE VALUE GATE MEASURED, and why its bound is 1e-6 rather than 1e-3.**
Fed `k...IdxQPost` and `k...IdxKRaw` — the oracle's own roped indexer query and
raw keys, captured by intercepting `apply_rotary_pos_emb` and slicing
`index_qk_proj`'s output — the composed logits are BIT-IDENTICAL to the oracle's
own pre-top-k `scores`: max abs **0** over a scale of 3.365 (12 logits,
sub-budget) and 6.239 (60 logits, over-budget). The reassociation the spec names
(upstream divides AFTER the head sum, the op's fold multiplies BEFORE it) does
not move a value at these shapes. Fed this port's OWN bf16 projection and bf16
RoPE instead, the same comparison lands at 1.76e-3 and 1.50e-3 — measured, not
feared, and it is why the fixture captures the oracle's inputs at all. The two
are separate cases: one gates the CONSTANTS, the other gates the INPUTS
(measured 0 and 3.8e-3 / 4.2e-3 against a bf16-sized 1e-2), and neither pretends
to be the other.

**THE OTHER MEASURED MARGINS**, so that a later reader can see which bound is
load-bearing and which is slack. Block output vs the oracle: 5.05e-3 and 5.58e-3
against 3e-2 (5.4x). The decode step: 6.13e-3 against 3e-2. The roped indexer
query: 4.22e-3 against 1e-2, a 2.4x margin and the tightest in the file — one
bf16 ulp is the floor there, because `vt::RopeFromCache` computes each rotated
pair in f32 and stores once where upstream multiplies and adds in bf16.

**WHAT THE CONTEXT LENGTH BUYS, measured again at the BLOCK layer.** At kv_len
3002 with the released indexer values (budget 2048, compress ratio 4) the block
reads `keys_visited` **16400** against a dense **24016** — 512 blocks x 4 rows
plus the 2-token ragged tail, x 4 query heads x 2 softmax passes. Below 2048
every candidate is selected and that assertion is trivially true, which is why
the case exists at 3002 and not at the fixture's 23.

**THE NaN PROBE AT THE BLOCK LAYER, and what it does and does not add.** The
block is run twice over identical inputs, the second time over a cache whose 12
of 23 rows the block's OWN selection does not name are bf16 NaN in both K and V
(11 attended). The output is finite AND bit-identical to the clean run, and
`keys_visited` agrees. That convicts a mask at the BLOCK's consumer call, which
the op-layer probe cannot do because it does not know what the block selects.
What it does NOT add is the fetch-level property — that the bytes were never
READ — which stays discharged by the `mprotect(PROT_NONE)` probe in
`test_qwen4_exp_qsa_device.cpp`; M20 is what says this block's only consumer call
is the op that probe covers.

**THE FRESH-REVIEW REPAIR: A HEADER THAT CLAIMED A GATE NOBODY HAD WRITTEN.**
The header said of the two rope layouts "the caller builds both from one table
and the gate asserts they agree". Nothing asserted it. The test's own `BuildRope`
DERIVES `packed` from `cos`/`sin`, so the two agree BY CONSTRUCTION in every case
in the file — which is not an assertion, and a layer loop that built them
inconsistently would diverge silently while the header told the next reader it
was covered. The block accepted two independently supplied layouts and
cross-checked neither.

Repaired by writing the check rather than softening the sentence, because a
silent divergence in a layer loop is the failure this campaign keeps finding.
`CheckRopeLayoutsAgree` refuses unequal heights, then compares a BOUNDED SAMPLE
of rows — row 1, the midpoint and the last row — value for value at one bf16 ulp
(2^-7); it refuses a non-CPU-resident pair BY NAME rather than skipping, because
a check that silently does not run on a device arm is a mute switch. Row 0 is not
a probe: cos is 1 and sin is 0 at every frequency there, so it agrees under every
difference the probe exists to catch. A full comparison is not paid, and that is
a cost decision rather than an oversight: it would be O(P * rot) per QSA layer per
step to re-check a constant.

**THE RED CAME FIRST, AND IT MEASURED THE SILENCE.** Two subcases were added
BEFORE the check existed. One perturbs the FULL `cos` table at row `c.seq - 1`
— a row nothing else in the block reads, because `vt::RopeFromCache` reads the
PACKED cache and the compressor reads only block-start rows (multiples of 4, and
10 is not one at seq 11); the other hands the two layouts different heights. On
the pre-check head, build rc 0, both reported **did NOT throw at all** and the
other **2829 of 2831 assertions passed** — so the divergence was invisible to
every value gate in the file, which is the claim the header had been making in
reverse. With the check in place: **8 of 8 cases, 2831 of 2831 assertions**. M22
above deletes the production call site on the repaired head and reds both
subcases again, which is what says the subcases gate the CHECK and not the
arithmetic.

**M1 AND M13 WERE RE-ARMED ON THE REPAIRED HEAD**, because a later commit can
silently disarm an earlier commit's mutation proof. The table above was measured
against the pre-repair file (digest `ab132cafcd327dda…`); the repaired file is
`e837cf290a86bd0d…`. M1 (inherit `n_head ** -0.5`) reds 1 of 8 / 2 assertions and
M13 (the indexer key at row 0) reds 1 of 8 / 2 assertions, both unchanged from
their recorded rows. The remaining rows are not re-measured, and the reason is
stated rather than assumed: the added check reads ONLY the two rope operand
tables and compares them with each other, and no mutation in the table alters
what a caller passes in those two arguments.

**NO REACHABILITY MUTATION IS AVAILABLE AT THIS LAYER, and that is the honest
statement rather than an omission.** AGENTS.md `## Nothing lands dead` wants a
production call site deleted, and there is none to delete: `## Owed` records this
block as unreached with the row and the issue that own the wiring. M20 is the
strongest statement this slice can make — it proves the tests enter the consumer
through `vt::Qwen4ExpQsaGatherAttention` and not through some other path.
## Mutation record — W5c-1 (#2031)

Every mutation was sha256-proven applied, **its BUILD rc was read before any
test result**, the tree was restored byte-for-byte with the hash re-checked, and
the final head was re-measured green afterwards. `runner.cpp` was measured at
`e538172d207f…`, `qwen4_exp_registry.cpp` at `2c30140e7b65…`,
`qwen4_exp_gguf_weights.cpp` at `b88f6e9ba247…`, and all three hashes are the
head's. I is the one row whose CASE gained assertions after its first run (the
`KVBytesPerBlock` pair), so it was re-run on the final head and reads the same
2 cases / 5 assertions.

### The RED, before the change

`test_qwen4_exp_kv_cache` at the branch base, every case entering through
`reg.factory->make_kv_cache`:

```
test_qwen4_exp_kv_cache.cpp:131: ERROR: test case THREW exception:
  Qwen4ExpForConditionalGeneration: the KV-cache spec is not ported yet
  (W4 owes the QSA indexer side cache and W2 the third conv state for the
   n-gram token history). See .agents/specs/qwen4-exp-flash-next.md and #1978.
[doctest] test cases:  3 |  0 passed | 3 failed | 0 skipped
[doctest] assertions: 32 | 23 passed | 9 failed |
```

Green at the head: **4 cases / 399 assertions / rc 0** (the fourth case, the
`--kv-cache-dtype fp8` consequence, was written after the first red).

### Counts, before and after, on the same tree

| Suite | Before | After |
|---|---|---|
| `test_runner` | 31 / 884 / rc 0 | 32 / 990 / rc 0 |
| `test_qwen4_exp_kv_cache` | did not exist | 4 / 399 / rc 0 |
| `test_qwen4_exp_gguf_weights` | 11 / 2970 / rc 0 | 11 / 2975 / rc 0 (one new SUBCASE inside an existing case) |
| `test_qwen4_exp_scaffold` | 12 / 296 / rc 0 | 12 / 296 / rc 0 |
| `test_qwen4_exp_qsa` | 14 / 7263 / rc 0 | 14 / 7263 / rc 0 |
| `test_qwen27_paged_forward` | 31 / 770 / rc 0 | 31 / 770 / rc 0 |
| `test_nemotron_h_paged_forward` | 13 / 3269 / rc 0 | 13 / 3269 / rc 0 |
| `test_kimi_linear_paged` | 8 / 206 / rc 0 | 8 / 206 / rc 0 |

`test_runner` moves by exactly the one case this wave adds. `test_qwen4_exp_qsa`
is byte-identical although its header changed, which is the check that the
#2198 fix touched only comments.

### The battery

| # | Mutation | Build | Result |
|---|---|---|---|
| A | delete `alloc_recurrent_layer_states` **inside `if (multi_cache_topology)`**, in its `membership_by_name && has_mamba_group` recurrent loop | rc 0 | `test_runner` RED — and ONLY the new case, confirmed scoped: `1 case / 0 passed / 1 failed / 31 skipped`, at `REQUIRE(runner.gdn_state().size() == 3)`. The three model suites stay byte-identically green. **This is the `## Owed` item `.agents/specs/recurrent-multistate.md` recorded: at that row's head the same deletion left ALL FOUR suites fully green** |
| B | **CONTROL** — delete the LEGACY single-topology `is_gdn` call site | rc 0 | `test_runner` rc 139 (10 of 13 reached cases failed), `test_nemotron_h_paged_forward` rc 139 (5 of 5 reached), `test_kimi_linear_paged` rc 1 (2 of 8), `test_qwen27_paged_forward` 31 / 770 / rc 0. The deletion harness is LIVE, so A's scoped red is a finding and not a dead instrument |
| C | the recurrent alloc AND view read `state_dtypes[i < 2 ? i : 0]` — states 2 and 3 get `dtypes[0]` | rc 0 | `test_runner` RED, 2 cases. Scoped to the new case: 13 of 106 assertions, every one on `states[3].dtype`, `states[3].Bytes()`, or a total that sums it. The three model suites stay green |
| D | the recurrent view reads `state_shapes[i == 2 ? 1 : i]` — state 2 gets the TEMPORAL shape | rc 0 | `test_runner` RED, 2 cases. Scoped: 16 of 106, on `states[2].rank`, its shape, its bytes and the two byte-identity totals. The three model suites stay green |
| E | publish group 2 as a `FullAttentionSpec` instead of an `MLAAttentionSpec` | rc 0 | `test_qwen4_exp_kv_cache` RED, 2 of 4 cases: the `kMlaAttention` kind, the `MLAAttentionSpec` downcast, and BOTH `fp8` refusal assertions — because a non-MLA third group is one an fp8 cache would silently accept |
| F | delete the `block_size % compress_ratio` refusal | rc 0 | `test_qwen4_exp_kv_cache` RED, 4 assertions, all in the refusal case. Nothing else moves |
| G | delete the non-uniform `attention.compress_ratios` refusal in `Qwen4ExpHfConfigFromGguf` | rc 0 | `test_qwen4_exp_gguf_weights` RED, 2 assertions. **Before this wave the same deletion left that suite fully green** — the refusal existed and gated nothing |
| H | **REACHABILITY** — unhook `.make_kv_cache` from `kQwen4ExpFactory` | **rc 1** | **A BUILD REFUSAL, not a test verdict, and it is read as such:** `error: 'MakeQwen4ExpKVCache' defined but not used [-Werror=unused-function]`. The production factory table is the function's ONLY reference in the tree, so the compiler proves the reach that a test result would only have suggested. No suite ran under this mutation |
| I | drop the `number_of_conv_states() == 3` branch, so the group always publishes two states | rc 0 | `test_qwen4_exp_kv_cache` RED, 2 of 4 cases: the four-shape `REQUIRE`, the 184336 B surcharge, the 51614080 B slack and the 3391504 B page. The uniform-cost accounting is load-bearing rather than decorative |

**Why A needed a NEW fixture and the existing one could not do it.**
`test_runner.cpp`'s "a multi-cache topology keeps its recurrent group" already
combines a multi-cache attention set with a mamba group, and it survives A
untouched: everything it asserts — `layer_kv_class_`, `gdn_group_id_`,
`recurrent_group_ids_`, the per-layer index lists — is computed BEFORE the
allocation loop runs. Classification and allocation are different failures, and
only the second one is what a short KV cache is.

**What the battery did NOT reach**, stated because a battery's silence is not a
result: the four-state group is never allocated on a DEVICE (the CPU host takes
`CacheBuffer`'s host-vector arm), nothing decodes through the published caches,
and no mutation here can see the zero-seeded n-gram history, because no test in
this tree reads that row's CONTENTS. All three are under `## Owed`.

## Mutation record — W5b-6 (#2218)

The gamma-polarity wave. Every mutation was sha256-proven applied, **its BUILD
rc was read before any test result**, the tree was restored byte-for-byte with
the hash re-checked, and both were RE-ARMED on the final head after the registry
comment landed. `cpu_qwen4_exp.cpp` was measured at `4accd54e82be…` and
`qwen4_exp_weights.cpp` at `81328de99cc1…`; both are the head's.

### The RED, before the change

`test_qwen4_exp_forward`, the new composition case, driven through
`ModelRegistry::Load` on the synthetic `qwen4exp` file:

```
tests/vllm/models/test_qwen4_exp_forward.cpp:222: ERROR:
  CHECK( MaxAbsDiff(mixed, want_mixed) < 1e-5f ) is NOT correct!
  values: CHECK( 1.50578 <  1e-05 )
  logged: site layer0.attn_hc
  ... identically at site layer0.mlp_hc and site model.mixer
[doctest] test cases:   1 |   0 passed | 1 failed | 0 skipped
[doctest] assertions: 409 | 406 passed | 3 failed |
```

1.50578 against a 1e-5 bound is not a tolerance question. `w_hf` is in [0, 1) on
this fixture and `1 + w_hf` in [1, 2), so the two parameterizations are a whole
multiplicative unit apart; on the RELEASED checkpoint `w_hf` sits within an ulp
or two of zero and the wrong one produces a stream scaled by ~0.

### Counts, before and after, on the same tree

The base was measured by checking `HEAD~1`'s copies of the four changed files
into this worktree, rebuilding (rc 0) and running, then restoring — not by
quoting the numbers a previous wave recorded.

| Suite | Before | After |
|---|---|---|
| `test_qwen4_exp_forward` | did not exist | 1 / 421 / rc 0 |
| `test_qwen4_exp_hc_device` | 9 / 87 / rc 0 | 9 / 87 / rc 0 |
| `test_qwen4_exp_hc` | 15 / 246 / rc 0 | 15 / 246 / rc 0 |
| `test_qwen4_exp_gguf_weights` | 11 / 2975 / rc 0 | 11 / 2975 / rc 0 |
| `test_qwen4_exp_ple_device` | 10 / 538 / rc 0 | 10 / 538 / rc 0 |
| `test_qwen4_exp_qsa_device` | 12 / 4697 / rc 0 | 12 / 4697 / rc 0 |

The op's numeric contract changed and **every existing count is identical**,
which is the check that the change is a re-parameterization and not a new
answer: the goldens store `w_hf` either way, the fold simply moved from the test
harness into the kernel. The fixture extraction is likewise count-neutral on the
loader suite, 11 / 2975 before and after.

### The battery

| # | Mutation | Build | Result |
|---|---|---|---|
| M-P1 | the kernel drops the `1 +`, i.e. the pre-#2218 contract restored | rc 0 | `test_qwen4_exp_forward` RED 1/1, at all three hyper-connection sites; `test_qwen4_exp_hc_device` RED 4/9, 12 of 87 assertions. The op half is gated |
| M-P2 | `LoadGatedResidual` stops unshifting, `unshift=false` | rc 0 | `test_qwen4_exp_forward` RED 1/1 **at its precondition**, after 6 assertions — the `model_gamma + 1 == file_gamma` `REQUIRE` fires before any arithmetic runs; `test_qwen4_exp_gguf_weights` RED 1/11, 25 assertions. The LOADER half is gated, so a future edit that moves the fold back into `load_weights` cannot land silently |

M-P2 is the half that matters. A case that only reddened on M-P1 would gate the
op against a number the test chose; reddening on both is what makes it a gate on
the SEAM.

### What the battery did NOT reach

Stated because a battery's silence is not a result.

- **The injection arm does not discriminate polarity at this fixture and the
  case says so out loud.** `2 * sigmoid(inject . normed / hc)` runs the
  fixture's `inject` ramp against a 128-wide normed row and reaches ~10^4 under
  BOTH gammas, so the sigmoid saturates at 2.0 either way. The case asserts the
  saturation explicitly, so the day it stops being saturated is loud rather than
  silent, and `mixed` carries the whole discriminating claim.
- **Nothing here decodes.** The composition gated is loader -> one op. The layer
  loop that would put 97 of these calls in sequence does not exist, so no token,
  no `hyper` stream and no `lm_head` is involved.
- **No CUDA arm was measured** because none exists; the op is CPU-only and the
  spec carries that under `## Owed`.

### The fresh review's findings, and what each one cost

The review returned `PASS` on the change: the mutations reproduce and all eight
pre-existing suites are count-identical. Six of its eight findings were prose,
records, a dead build define and a merge. The other two changed something
measured — one a published claim, one a dtype — and both are recorded here,
because a reader of this section would otherwise take the earlier text at face
value.

**The "first suite in this tree" claim was FALSE unscoped.** `## Now` said this
was the first suite here to load a gamma through `ModelRegistry::Load` and run it
through a device op in one case. `tests/vllm/models/test_nemotron_h_paged_forward.cpp`
and `tests/vllm/models/test_kimi_linear_paged.cpp` already do both inside a
`TEST_CASE`. Scoped to `qwen4_exp` the claim holds, and the argument it supports
— eleven single-sided waves of THIS row could not see the contradiction —
survives unchanged. Corrected in `## Now`, in `## Owed` and on #2218 itself.

**The four-gamma attribution was wrong about one CONSUMER.** The `## Owed` entry
said `RunQwen4ExpQsaBlock` normalizes all four QSA gammas through
`vt::RmsNorm(gemma = true)` and then cited three line pairs. The count exposed
it: `idx_k_norm` never reaches `vt::RmsNorm`. It goes to `Qwen4ExpQsaIndex`
(`qwen4_exp_qsa_block.cpp:401-403`) and is consumed by `vt::Qwen4ExpQsaCompress`
(`:181`), which adds the 1 itself. Same polarity, different op, so "three of the
four consumers already add the 1" stands with the consumer named correctly.

**THE FOLD'S DTYPE HAD DRIFTED, AND THE BAND WAS ABSORBING IT.** Before this
wave, the wide-accumulator case handed one identical `float` multiplier to both
arms. After it, the kernel folded `1.0f + w` in f32 while the double reference
folded `1.0 + (double)w`, so the two arms no longer described the same multiplier
and the case's own comment — "the only thing this widens is the reduction" —
stopped being true. Nothing failed, which is the point. Measured on exactly the
data in the case, by forcing the bound to `1e-30` and reading the logged `worst`:

| Reference's fold | Worst absolute deviation, `mixed` vs reference |
|---|---|
| `1.0f + w_hf`, widened AFTER (f32, as landed here) | 1.17323e-06 |
| `1.0 + (double)w_hf` (the drifted form) | 9.8457e-07 |

Both sit far inside the band — the bound is `1e-5` and the `float ss` mutant
reads 6.702e-4 — so no tolerance was ever at risk. What was at risk is the
meaning of the number: **1.173e-06 is the figure this file and the W5b-2 table
record as "ours, double accumulator", and the drifted form no longer reproduced
it.** The f32 fold is also what upstream does —
`output * (1.0 + self.weight.float())` (`modeling_qwen4_exp.py:177`) folds a weak
Python `1.0` into an fp32 tensor and the promotion stays fp32 — so mirroring
upstream and restoring the recorded measurement are the same edit. AGENTS.md
"Inherit vLLM defaults" decides it either way: f32 is the default and the wider
value would have been the annotated exception, unannotated.

## Mutation record — W5d-2 (#2249 item 5)

The interleaved-mRoPE cos|sin table builder, `BuildMropeCosSinHost`. It was
`static` at `src/vllm/model_executor/models/qwen3_5.cpp:9472`, so the tables
Qwen4-Exp's QSA half of the layer loop needs could not be built from another
translation unit and the QSA block would have had to grow a second copy of the
axis selection and the angle math.

**WHICH SHAPE, AND WHY THE SIMPLER ONE.** `RunGdnBlockPaged` (W5b-1) and
`RunMoeBlock` both needed a thin PUBLIC WRAPPER over a private definition,
because their signatures name types qwen3_5.cpp declares privately
(`StepDevInputs`). This one names nothing private — `std::vector<int32_t>`,
`int64_t`, `vllm::HfConfig` — so the extraction is the `static` keyword and a
declaration in `include/vllm/model_executor/models/qwen3_5_mrope.h`. The
definition does not move and there is exactly ONE implementation: qwen3_5.cpp's
own two call sites now resolve through the same public declaration qwen4_exp
will use, which is what `AGENTS.md` `## Shared seams` requires and what a
copied second table builder would have broken.

**BYTE IDENTITY, TWICE.** First textually: `git show
94de63ff5:src/vllm/model_executor/models/qwen3_5.cpp | sed -n '9473,9514p'`
sha256s to `259b1b932cae0611ca6dbde4ad63214e0d1365efe3b708b8ef7d38a7894688f1`,
and so does the body on this branch — the whole diff to that function is the
`static` keyword and two comment lines. Second by VALUE, because the keyword
that changed is exactly the one that decides which definition a caller binds
to: `tests/vllm/models/test_qwen3_5_mrope.cpp` pins 152 f32 BIT PATTERNS across
four cases against what the FILE-STATIC produced at base SHA `94de63ff5`,
captured by compiling its `sed`-extracted text in a standalone harness. The
comparison is bitwise and not an epsilon — this is a pure host computation over
`std::cos`/`std::pow` with no reduction-order freedom, so a tolerance would hide
the one defect an extraction can introduce.

**Counts, before and after, on the same tree.** 26 pre-existing qwen3_5 /
qwen4_exp suites built and run at base and at head, identical exit status and
identical case and assertion counts on every one (`diff` of the two count files
is empty). The new suite adds 4 cases / 157 assertions. The population is every
`vllm_cpp_add_test` target in `tests/CMakeLists.txt` whose name matches
`qwen3_5`, `qwen35` or `qwen4_exp`, less the benchmark
`bench_qwen3_5_vl_tower` and less this wave's own `test_qwen3_5_mrope`.

**WHICH TREE THAT 26 WAS COUNTED ON, because merging `main` moved it.** The
count is base `94de63ff5` against branch head `c1ccbac19`, both of which
predate the merge of `main` in this branch. That merge brings in W5b's
`test_qwen4_exp_forward` ([#2031](https://github.com/mudler/vllm.cpp/issues/2031),
landed on `main` as `a6f933b81`'s neighbour), which makes the same glob match 27
targets on the merged tree. It is NOT a 27th row of this neutrality
measurement and cannot be: it existed at neither end of the before/after pair,
so it has no before. It is `main`'s own gate for `main`'s own wave. The 26 is
therefore a statement about the two trees named here and not about the merged
head, which is the distinction this section previously left for a reader to
make.

**FOUR of the 26 measure NOTHING on this host, and only one of them says so.**
An earlier revision of this section said "23 suites, two of which do not
measure". Both halves were wrong. Re-measured on this CPU-only host at this
head:

| suite | rc | cases | assertions | why it measures nothing |
|---|---|---|---|---|
| `test_qwen35_paged_engine` | 77 | — | — | prints `*** GATE NOT RUN — SKIPPED (exit 77), this is NOT a pass ***`; the Qwen3.5-0.8B snapshot at revision `2fc06364` is not cached here. **This is the one that is honest about itself** |
| `test_qwen35_gguf_spec_decode` | 0 | 3 | **0** | `SKIP: set VLLM_MTP_GGUF_MODEL` |
| `test_qwen3_5_vl_e2e` | 0 | 1 | **0** | `SKIP: Qwen3.6-27B checkpoint absent (set VLLM_QWEN36_CKPT)` |
| `test_qwen3_5_vl_video_e2e` | 0 | 1 | **0** | the same skip |

The last three exit 0 and print `[doctest] Status: SUCCESS!`. That is a skip
wearing a pass, and a count-diff over a population containing them is neutral by
construction on those three rows, so they carry no neutrality evidence at all.
They are listed so that a reader does not read 26 green suites as 26
measurements.

**AND THIS BOUNDS THE M3 REACHABILITY EVIDENCE, WHICH IS THE PART THAT MATTERS.**
`test_qwen3_5_vl_e2e` and `test_qwen3_5_vl_video_e2e` are the STRICT token-exact
end-to-end gates on `VLGenerateCoreGdn`, the shared driver core holding the two
production call sites M3 deletes. On a host that has the Qwen3.6-27B checkpoint
they would be the strongest witnesses M3 has. Here they measure nothing, so the
M3 red rests ENTIRELY on `test_qwen3_5_moe_vision` (7 cases / 38 assertions),
whose `qwen3_5_moe_vl_image_forward_uses_MRoPE_positions_not_plain_1d` is the
single case that goes red. One case, one assertion, is the whole reachability
proof on this host. `test_qwen3_5_moe_vision_hw` does not extend it either: it
measures 3 cases / 23 assertions but its own e2e case skips on
`VLLM_MOE_VISION_E2E`. This is a HOST condition and not a defect in the
mutation — it is stated because a reader on a GPU host with the checkpoint gets
strictly more evidence than this run produced, and a reader without it gets
exactly one assertion.

**Upstream.** No divergence found. The interleaved axis masks mirror
`vllm/model_executor/layers/rotary_embedding/mrope.py:60-63` at the parity pin
`5559679229` (`h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 *
mrope_section_h)`, and the `w` twin), `apply_interleaved_rope` (`:190-198`)
states the same layout as a tensor rewrite, and the chunked branch mirrors the
same function's `else` arm (`:66-70`). The per-pair frequency is
`base ** (-2 * pair / rotary_dim)`, which is `RotaryEmbeddingBase`'s inv_freq.

| # | mutation | build rc | target | result |
|---|---|---|---|---|
| M1 | swap the cos and sin stores for the `h` axis (`axis == 1`) inside the extracted function | 0 | `test_qwen3_5_mrope` | **RED, 4 of 4 cases, 42 of 157 assertions.** The first failures are index 1 and index 9 of C1 trading values, which is the swap seen directly |
| M2 | change the position offset by one (`positions3[axis * T + i] + 1`) | 0 | `test_qwen3_5_mrope` | **RED, 4 of 4 cases, 141 of 157 assertions.** The 16 survivors are the pairs whose frequency is small enough that the f32 store absorbs one position |
| M2b | `pair <= 3 * sec[1]` -> `pair < 3 * sec[1]`, the upstream `<=` | 0 | `test_qwen3_5_mrope` | **GREEN — an EQUIVALENT MUTANT, and provably so.** The two forms differ only at `pair == 3 * sec[1]`, and the guard already requires `pair % 3 == 1` while `3 * sec[1]` is divisible by 3, so no input separates them. Upstream's `<=` and a `<` are the same function here. Recorded rather than replaced by a stronger case, because the next reader will reach for this mutation too |
| M2c | shift the same boundary instead: `pair <= 3 * sec[1] + 1` | 0 | `test_qwen3_5_mrope` | **RED, 2 of 4 cases, 8 assertions.** This is the section boundary actually under gate: on C1 (`sec = {4,2,2}`, half 8) pair 7 flips from the `t` axis to the `h` axis |
| M2d | the SAME shift on clause TWO, the `w` axis: `pair % 3 == 2 && pair <= 3 * sec[2]` -> `... + 2`. `+2` and not `+1`, because `3 * sec[2]` is divisible by 3 and the clause already requires `pair % 3 == 2`, so `+1` would be a second equivalent mutant for exactly M2b's reason | 0 | `test_qwen3_5_mrope` | **RED, 1 of 4 cases, 2 of 157 assertions.** Added on review repair, because M2b's green is only honest if the OTHER clause's reachable boundary is shown to red too — otherwise a reader cannot tell an equivalent mutant from an ungated one. Pristine `qwen3_5.cpp` sha256 `0b4517b3246e6e49fd8b0fa3a8ad7adc5c39b2846a4800966733688fb0d8d9fe` before, `c00f7a461b65a3260ea255b30bc03864bd7ad53cfb9379905cc41fe38b10ff8f` under the mutation, and back to `0b4517b3…` on restore; BUILD RC 0 read before the test result on both legs; re-run green 4 of 4 cases / 157 of 157 assertions, and `test_qwen3_5_moe_vision` 7 of 7 / 38 of 38 |
| M3 | REACHABILITY: delete both production call sites in `qwen3_5.cpp` (`VLGenerateCoreGdn`'s prefill build and its decode-continuation build) and pass `nullptr` for the cache | 0, after a `(void)` for `-Werror=unused-parameter` | `test_qwen3_5_moe_vision`, `test_qwen3_5_mrope` | **`test_qwen3_5_moe_vision` RED on exactly `qwen3_5_moe_vl_image_forward_uses_MRoPE_positions_not_plain_1d` (1 of 7 cases, 1 of 38 assertions)** — the VL greedy driver reaches the extracted function and a test enters through the driver. It is NOT a production entry point; see the paragraph below. **`test_qwen3_5_mrope` stays GREEN, and it must:** it is a unit and seam case that calls the function directly, so it measures the function and never that anything reaches it. Stated here rather than left to be inferred |

**WHAT M3 DOES NOT PROVE, MEASURED RATHER THAN ASSUMED.** The chain M3 reds
stops one hop short of a production entry point, and this wave did not create
that and does not close it. `grep -rn 'Qwen3_5MoeVLGenerateGreedy|Qwen3_5VLGenerateGreedy'`
over `src/ include/ examples/ tools/ benchmarks/` returns the four DEFINITIONS
in `src/vllm/model_executor/models/qwen3_5.cpp:9892,9915,9960,9974` and their
six declaration lines in `qwen3_5.h` / `qwen3_5_dense.h` — and NOTHING else.
Every CALLER is in `tests/`. The registered factories for
`Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration`
carry no multimodal hook, so `ModelRegistry::Forward` cannot arrive here, and
`include/vllm/entrypoints/openai/chat_mm.h:266-267` already says so in the tree's
own words for the sibling Qwen3-VL driver: the greedy VL drivers run "outside
`ModelRegistry::Forward`". So `BuildMropeCosSinHost` is reached by a public,
gated, non-test caller, and that caller is not yet routed from
`include/vllm.h`, the loader, `ModelRegistry::Forward` or a server path. The
extraction changes nothing about that either way — the function had exactly this
reach before the `static` came off — and it is recorded under `## Owed` rather
than left for a reader to discover, because `AGENTS.md` `## Nothing lands dead`
asks the question at every merge and silence is not an exception.

Every mutation was applied to a pristine `qwen3_5.cpp`, sha256-proven applied,
rebuilt with the BUILD RC read BEFORE any test result, run, then restored from a
byte-identical copy and re-proven at
`0b4517b3246e6e49fd8b0fa3a8ad7adc5c39b2846a4800966733688fb0d8d9fe`, rebuilt and
re-run green. M3's first attempt did NOT build — deleting the call leaves
`pos3_prefill` unused under `-Werror=unused-parameter` — which is the W5b-1
mutation-B trap again and the reason the build rc column is in this table.

## Mutation record — W5d-1 (#2249 item 1)

`vt::RmsNormGroup` / `OpId::kRmsNormGroup`, the ungated per-group RMS norm the
PLE half of the layer loop needs and the one primitive `include/vt/ops.h` named
as missing in its own words.

**THE TREE EVERY NUMBER BELOW WAS MEASURED ON**, because an evidence table that
does not name its tree is not evidence. The RED, the green, the six mutations
and the suite counts were all taken at base `94de63ff5`. The branch was then
rebased forward twice as `main` moved under it, onto `6e805abcf` (`QUANT-EXL3`
W3 — `cpu_exl3_kernels.cpp`, `cuda_exl3.cu`, `test_exl3_gemm.cpp`,
`dense_weight_loaders.h`) and then onto `5f8a70705` (`SPEC-DFLASH2` #2252 —
`qwen3_dflash*`). Neither touches a file this op compiles against.
`test_ops_rms_norm_group` was rebuilt and re-run on that head: **build rc 0,
7 cases / 69 assertions / rc 0**, unchanged. The mutation battery was NOT re-run
after either rebase, and that is stated rather than implied.

**The review repair then MERGED `origin/main` into the branch** rather than
rebasing a third time, and has now had to do it THREE TIMES, because `main` kept
moving while the repair ran. `scripts/agent-preflight.sh` skips both trailer
gates whenever `origin/main` is not an ancestor of HEAD — "this branch is behind
it and the trailer gates did NOT run" — which had quietly voided the review's
own `PREFLIGHT_RC=0`. The commits merged in are `1bc16ca3c`
(`PERF-LAGUNA-GROUPED-GEMV` spec) and `37fbccea8` (`MODEL-TEXT-GLM-MOE-DSA` spec)
first, then `fa9903b86` (`LTX25-ORACLE-ABSOLUTE`, #2210), and finally `3ed2378a3`
(W5d-2, #2249 item 5, via #2264). THE FIRST TWO touch `.agents/`,
`docs/USAGE.md` and two `scripts/` files and NO compiled input: `git diff
--name-only` over that delta returns nothing under `src/`, `include/`, `tests/`
or `third_party/`, and `ninja` answered "no work to do" after it.

**THE THIRD MERGE IS DIFFERENT, WHICH IS WHY THE SENTENCE ABOVE IS SCOPED TO THE
FIRST TWO RATHER THAN LEFT TO COVER ALL THREE.** The delta `fa9903b86..3ed2378a3`
is fourteen commits and it DOES move compiled input, including both files this
wave itself edits: `include/vt/ops.h`, where W5b-6 rewrote the
`vt::Qwen4ExpGatedResidual` contract comment in place (#2218 via `a6f933b81`),
and `tests/CMakeLists.txt`, where W5d-2 registered `test_qwen3_5_mrope`. Git
merged both without a conflict and both merges are purely ADDITIVE against
`main`: `git diff origin/main` over the two files shows this wave's blocks and
nothing removed. The one consequence a conflict-free merge could still have hidden
is an id shift, because `kRmsNormGroup` is appended before `kCount` and so is
every other new op — nothing on `main` appended an enumerator, and the merged
header compiles to `kRmsNormGroup == 140`, `kCount == 141`, so the `id 140` the
refusal prints below is still this op's id. `.agents/specs/qwen4-exp-flash-next.md`
did conflict and was resolved as a SET against the merge base rather than by
taking either side, and the one anchor the merge moved is corrected under
`## Owed`: the `kTENSTORRENT` `RegisterOp(OpId::kRmsNorm, ...)` line is at
`tenstorrent_ops.cpp:5323` on this head, not the `:5216` recorded before
`BACKEND-TENSTORRENT-QWEN35` W5/W6 landed.

**A false alarm is recorded here rather than buried, because it nearly landed a
duplicate.** `origin/main` is a shared ref in a shared checkout, and another
session fetched it mid-repair. Comparing the merged index against the ref AFTER
that fetch read as though the first merge had silently dropped its tail row
(#2220), and a commit was written to "restore" it. The merge had dropped nothing:
the row arrived with `fa9903b86`, which had not been merged yet. Appending it by
hand would have produced a SECOND copy of that row on `main` — the duplicate
`check-agent-record.py` refuses and `check-issue-index-append-only.py` will not
permit anyone to remove. The commit was dropped before it left the worktree.
`check-issue-index-append-only.py` returned rc 0 on BOTH the pre- and
post-"restore" heads, so the gate would not have caught it either way: the
control that worked was diffing the row-number list against the ref and asking
WHICH commit authored the row, not the checker.

At the THIRD merged head, from a build directory configured from scratch, build
rc 0 read before any test result, **the same FIVE of the seven suites below were
re-run and all five still match byte-for-byte**: `test_ops_rms_norm_group`
7 / 69, `test_ops_mamba2_gated_norm` 9 / 2107, `test_ops_glue` 13 / 115,
`test_qwen4_exp_hc` 15 / 246 and `test_qwen4_exp_hc_device` 9 / 87. Because that
merge brings compiled input with it, `main`'s OWN two new gates were built and
run here as well, and both reproduce the counts `main`'s own sections record:
`test_qwen4_exp_forward` 1 / 421 (W5b-6, #2218) and `test_qwen3_5_mrope` 4 / 157
(W5d-2, #2249 item 5). `test_qwen4_exp_scaffold` was re-run as well, at
12 / 296 / rc 0, because this merge EDITS the production refusal string that
suite's `SUBCASE("the forward")` pins — see `## Now` for why the string had to
change and which five substrings the suite holds. `test_qwen4_exp_ple` and
`test_qwen4_exp_ple_device` were NOT re-run at any merged head, and their rows
below still name `94de63ff5`.

**M4 and M5 were re-run at the SECOND merged head**, because the `## Owed`
sentence that repair corrects is a claim about exactly those two: M4 build **rc 1** with
`error: 'RmsNormGroupKernel' defined but not used [-Werror=unused-function]` and
NO suite run; M5 build **rc 0**, suite **rc 1**, **6 of 7 cases**, six throws of
`vt: no kernel for op RmsNormGroup (id 140) on device cpu (type 0)` raised at
`src/vt/op_provider.cpp:589`. `src/vt/cpu/cpu_ops.cpp` was restored
byte-for-byte after each, sha256 `e4a924b4…04b5` both times, rebuilt at rc 0 and
re-run green at 7 / 69. **They were NOT re-run at the third merged head**, so the
battery has not been re-measured since `main` began moving compiled input under
this branch; the throw site `src/vt/op_provider.cpp:589` and the printed
`id 140` were both re-checked there, the battery itself was not. Every other
number in this section still names `94de63ff5` and is not restated as if it were
measured here.

Method as in the sections above:
one textual change applied to a pristine tree, proved applied by a **sha256 that
moved**, the file `touch`ed so ninja cannot skip the rebuild, the **BUILD RETURN
CODE READ BEFORE ANY TEST RESULT**, then restored from a byte-identical copy and
`sha256sum`-verified against the pre-mutation digest. Suite
`tests/vt/test_ops_rms_norm_group.cpp`, **7 cases / 69 assertions / rc 0** green
at the head this table was measured on.

**WHY IT IS A NEW OpId AND NOT A FIELD ON `RmsNormArgs`.** `kRmsNorm` is
registered on more than one backend. A `group_size` added to its shared args
struct is IGNORED by every kernel not taught to read it, so a device whose
kernel was not updated would answer a grouped request with a whole-row norm —
no crash, no refusal, a plausible tensor. A separate OpId cannot fail that way:
an unregistered device refuses BY NAME, which M5 below measures. `kRmsNormGatedGroup`
is the in-tree precedent for exactly this split.

### The RED, before the change

The test file written first, against the tree at `94de63ff5`, compiling the test
translation unit alone (`ninja tests/CMakeFiles/test_ops_rms_norm_group.dir/vt/test_ops_rms_norm_group.cpp.o`):

```
BUILD_RC=1
test_ops_rms_norm_group.cpp:75:11: error: 'RmsNormGroupArgs' has not been declared in 'vt'
test_ops_rms_norm_group.cpp:120:7: error: 'RmsNormGroup' is not a member of 'vt'; did you mean 'RmsNormGated'?
```

The compiler's own suggestion is the gap in one line: the nearest thing this tree
had is the op that cannot express it. That red is a BUILD refusal and is read as
one — it says the op is absent, not that any arithmetic is wrong. The red for the
intended REASON is M1, which builds clean and fails on values.

### The gate, and why each half of it discriminates

The correctness assertions run against `tests/vllm/models/qwen4_exp_hc_goldens.inc`,
whose `k{A,B,C,D}_normed` arrays are `normed = mod.hc_norm(hyper)` — the pinned
oracle's OWN `Qwen4ExpTextRMSNorm(hc*hidden, group_size=hidden, eps)` output over
its own RAW gamma, dumped by `scripts/gen-qwen4-exp-hc-goldens.py` from
transformers **v5.16.0** (`modeling_qwen4_exp.py:158-181`, sha256
`77fec77d…c459`). Nothing in the correctness path is transcribed. A local
double-precision reference exists in the file, but ONLY to measure the
separations below; the op is never asserted against it.

| Defect | Separation from the oracle | kTol |
|---|---|---|
| reduce over the ROW, not the group | A 1.232, B 0.984, C 0.404, D 0.913 | 1e-5 |
| drop the `+ 1` on the gamma | A 2.279, B 2.181, C 2.053, D 1.986 | 1e-5 |
| drop eps | A 4.1e-6, B 1.67e-5, C 5.1e-7, **D 2.56e-2** | 1e-5 |

**The eps row is the reason case D exists and the reason an eps probe run at
A's scale is a mute switch.** At `hyper_scale = 1.7` the mean square is O(1) and
an eps of 1e-6 moves the answer by less than the tolerance; at D's
`hyper_scale = 0.01` it is 1% of the mean square. The file asserts BOTH
directions — `sep > 1e2 * kTol` at D and `sep < kTol` at A — so the fixture's
blind spot is recorded executably rather than left for the next reader to
rediscover.

### The battery

| # | Mutation | Build | Result |
|---|---|---|---|
| M1 | `RmsNormGroupKernel`: `group_size = h`, i.e. reduce over the whole row | rc 0 | **RED**, `7 cases / 5 failed`, `69 assertions / 26 failed`. Every value case moves: the oracle case at all four goldens, the four-orders-apart case, the fold case, the eps case and the bf16 rounding case. The two survivors are the two that call no op — the fixture-separation case and the refusal case |
| M2 | drop `if (args.gemma) wj += 1.0f`, the #2218 defect | rc 0 | **RED**, `4 cases failed`, `22 assertions failed`. This is the polarity the whole architecture now shares: every gamma is stored RAW and every consumer adds the 1, `ssm_norm` excepted |
| M3 | drop `+ args.eps` from inside the rsqrt | rc 0 | **RED**, `4 cases failed`, **`5 assertions failed`** — and the small count is the finding, not a weakness. Only goldens B and D move; A and C are BELOW the tolerance, exactly as the table above predicts. A probe placed only at A would have reported this mutation as survived |
| M4 | delete `RegisterOp(OpId::kRmsNormGroup, DeviceType::kCPU, ...)` | **rc 1** | **A BUILD REFUSAL, not a test verdict, and read as such:** `error: 'RmsNormGroupKernel' defined but not used [-Werror=unused-function]`. That registration is the kernel's ONLY reference in the tree, so the compiler proves the link a test result would only have suggested. No suite ran under this mutation |
| M5 | register the same kernel on `DeviceType::kCUDA` instead — the runnable form of M4 | rc 0 | **RED**, `6 of 7 cases` threw `vt: no kernel for op RmsNormGroup (id 140) on device cpu (type 0)`. This is the load-bearing reachability proof at the layer that exists: the suite reaches the kernel THROUGH `GetOp`, not by calling it directly, and the `op_provider.cpp` name entry is live too, because the refusal prints the op by name |
| M6 | delete the `args.group_size >= 1` refusal in the dispatcher | rc 0 | **RED**, rc 136 — `SIGFPE`, an integer divide by zero at `h / group_size`. The refusal is therefore load-bearing rather than decorative, and the default `group_size = 0` is genuinely unusable rather than quietly meaning "the whole row" |

M1-M5 target `src/vt/cpu/cpu_ops.cpp`, M6 `src/vt/ops.cpp`. M6 was re-measured on
the FINAL head after its refusal message was corrected; M1-M5 were measured on a
head that differs from the final one only in that message string, in a file they
do not touch.

### Counts on this head

| Suite | Result |
|---|---|
| `test_ops_rms_norm_group` | **7 / 69 / rc 0** (new; re-run identical on the rebased head) |
| `test_ops_mamba2_gated_norm` | 9 / 2107 / rc 0 |
| `test_ops_glue` | 13 / 115 / rc 0 |
| `test_qwen4_exp_hc` | 15 / 246 / rc 0 |
| `test_qwen4_exp_hc_device` | 9 / 87 / rc 0 |
| `test_qwen4_exp_ple` | 9 / 395 / rc 0 |
| `test_qwen4_exp_ple_device` | 10 / 538 / rc 0 |
| `test_qwen4_exp_qsa` | 14 / 7263 / rc 0 |
| `test_qwen4_exp_qsa_device` | 12 / 4697 / rc 0 |
| `test_qwen4_exp_qsa_block` | 8 / 2831 / rc 0 |
| `test_qwen4_exp_kv_cache` | 4 / 399 / rc 0 |
| `test_qwen4_exp_scaffold` | 12 / 296 / rc 0 |
| `test_qwen4_exp_gguf_weights` | 11 / 2975 / rc 0 |
| `test_qwen4_exp_gguf_load_plan` | 10 / 7462 / rc 0 |

**The BEFORE column is stated rather than re-measured, and the reason is
checkable.** `git diff --numstat` over `include/` and `src/` is `118/0`, `52/0`,
`2/0`, `31/0` — **zero deletions, zero modified lines**: a new enumerator before
`kCount`, a new args struct, a new function-pointer alias, a new declaration, a
new kernel with its registration, and a new name case. No existing behaviour is
reachable from any of it. Four of the rows above are additionally cross-checked
against numbers this spec already recorded before this wave — `test_qwen4_exp_qsa`
14 / 7263, `test_qwen4_exp_qsa_block` 8 / 2831, `test_qwen4_exp_kv_cache` 4 / 399
and `test_qwen4_exp_scaffold` 12 / 296 — and all four match exactly.

**What the battery did NOT reach**, because a battery's silence is not a result:
no CUDA arm exists to mutate; no production entry point calls the op, so no
mutation here can measure a reach that does not exist (`## Owed`); and the group
widths exercised are 4, 5 and 6, not the 2560 the released config uses, so the
f32 sum-of-squares accumulator is gated at toy width only.

## Mutation record — W5d-3 (#2249 item 2)

The wave that gave the QSA consumer a PAGED read path. Measured on an
`origin/main` base of `fa9903b860`, CPU only, Release, `-j 2`. The build return
code was read BEFORE any test result on every row, because a failed build reads
as a passing test.

**The instrument.** The paged cases fill the WHOLE flash cache with bf16 NaN
before the block runs, so every row a correct read never addresses — an unnamed
physical page, or the unused tail of the last named one — is not a number. That
is the same discriminator the W5b-4 gather-vs-mask case uses, doing a second job:
there `0.0f * NaN` convicts a MASK, here it convicts a wrong ADDRESS. The block
table is `{5, 3, 7}` against a logical `{0, 1, 2}`, sharing no fixed point, so the
three pages an identity-reading body touches are exactly three that are never
written. **An identity block table would make the whole case prove nothing**, and
that is why the permutation is stated here rather than left to the fixture.

**And one finding about the harness itself, measured rather than feared.**
`MaxRelDiff` folds with `std::max`, and `std::max(x, NaN)` returns `x`. So the
first RED capture below came back ALL NaN and the relative bound still printed
`0` and PASSED. A tolerance cannot see a NaN. The two paged cases therefore assert
FINITENESS FIRST and the oracle bound second, and the test says why.

| # | Mutation | Site | Build rc | Result |
|---|---|---|---|---|
| RED | the pre-W5d-3 body: `RowBase` always returns the CONTIGUOUS address `(p * HKV + kvh) * DH`, i.e. the paged arm reads slots linearly | `cpu_qwen4_exp_qsa.cpp` | 0 | **RED, 2 of 11 cases, 1537 assertions.** `CHECK(std::isfinite(v))` x1472 (every output NaN), `CHECK(differing == 0)` at `paged vs contiguous differing bf16 words 1472 of 1472`, and 64 more on the decode case. This is the capture of the gap #2249 item 2 names |
| M1 | OFF-BY-ONE in the page-table read: `pages[(p + 1) / page_size]` | `cpu_qwen4_exp_qsa.cpp` | 0 | **RED, 2 of 11 cases, 195 assertions.** `rel 0.309431 < 0.03` against the oracle, `differing 703 of 1472`, decode `rel 0.667465`. Note it is NOT all-NaN: an off-by-one lands on a WRITTEN page most of the time, which is precisely why the value comparison and the bit-exact one both have to be there |
| M2 | the partial final page read at FULL page length: the ragged tail runs to `ceil(kv_len / page) * page` instead of `kv_len` | `cpu_qwen4_exp_qsa.cpp` | 0 | **RED, 2 of 11 cases, 515 assertions.** 448 NaN outputs (row 7 of the last page is the one row the 23-token sequence never writes), `rel 1.24777`, `differing 1339 of 1472`, and the `keys_visited` equality with the contiguous arm |
| M4 | the PAGED STORE deleted (`dense_attn::WriteKvCache` never called) | `qwen4_exp_qsa_block.cpp` | 0 | **RED, 2 of 11 cases, 1537 assertions**, all NaN. The store site is gated, not merely present |

Every mutation was sha256-proved applied, and the tree was restored byte-for-byte
after each (`cpu_qwen4_exp_qsa.cpp` back to
`d95eea49e1800a25fb0b920a43c92936973c5e51e576651fa74400064a2497cc`,
`qwen4_exp_qsa_block.cpp` to
`feb0eccd41d39a1588a9ffd32db9bfec50ac01dc40e44e494ad426d7ca3c43b7`).

**M3, the reachability mutation, HAS NO SITE, and that is the finding rather than
an omission.** `.agents/reachability.md` asks for the production call site to be
deleted in a scratch copy. `grep -rn 'RunQwen4ExpQsaBlock\|Qwen4ExpQsaPagedCaches'
src include examples` returns only the block's own header and translation unit:
there is no production caller to delete, because
`ForwardQwen4ExpForConditionalGeneration` still refuses by name and the layer loop
is unwritten. So what these mutations measure is a CAPABILITY of the block, not
that anything reaches it. `## Owed` records the wave as UNREACHED with the owning
row and the issues.

**What the battery did NOT reach.** The device (CUDA) arm of the address mode does
not exist, so nothing here says a GPU resolves a page the same way. Nothing decodes
a real checkpoint through the paged arm. And no mutation here can see a wrong
INDEXER side-cache address, because that cache is still contiguous — #2249 item 3,
owed as W5c-2.

## Mutation record — W5c-2 (#2249 item 3)

The group-2 block-table gather. `GPUModelRunner::gather_block_table` had three
call sites and reached exactly two group ids, so a model publishing a THIRD
group — `qwen4_exp`'s QSA indexer side cache, an `MLAAttentionSpec` at
`compress_ratio` 4 — had that cache allocated and no map from a logical position
into its pages.

**WHAT WAS MIRRORED, AND WHY IT IS A LOOP AND NOT A THIRD NAMED ID.** Upstream
has no "the two special groups" shape at all. Its per-group metadata build runs
over `enumerate(kv_cache_groups)` and hands each group its own table —
`cm.block_table_tensor = _get_block_table(kv_cache_gid)`,
`vllm/v1/worker/gpu_model_runner.py:2551-2567` at the parity pin
`5559679229` — where `_get_block_table` (`:2318-2334`) is
`self.input_batch.block_table[kv_cache_gid].get_device_tensor(...)`, which is
byte-for-byte what this tree's `gather_block_table` does. Group 0 is gathered
once BEFORE the loop (`block_table_gid_0 = _get_block_table(0)`, `:2337`) and
carried into every iteration on `cm_base`; the loop body itself is guarded by
`if kv_cache_gid > 0:` (`:2565`), so upstream does NOT re-gather group 0 inside
it and this tree does. That is an OPTIMISATION on upstream's side, not a second
convention for "which groups are special": the value is the same table either
way. We pay one extra copy of a table the step already built rather than
special-casing an index, and the extra copy is disclosed here and in the commit
body. `GPUModelRunner::gather_group_block_tables` is that loop.

**WHERE THE TABLES GO, AND WHY NOT ON `CommonAttentionMetadata`.** Upstream fans
its per-group metadata out BY LAYER NAME (`:2551-2552`, "make layers in the same
group share the same metadata"), and this tree's mirror of that key is
`MultiKvCacheIndex` (KV-DSV4-MULTICACHE W3, #2068) — the channel that already
carries which group each published cache came from. The two new vectors are
indexed by GROUP ID rather than parallel to `attn_kv`, because a block table
belongs to a group and every layer in it shares one. `CommonAttentionMetadata`
carries exactly one table, the target group's, and widening it would put a
multi-cache field on every uniform step.

**WHAT IS STILL NOT REACHED, STATED BEFORE THE BATTERY.** The gather runs on the
production `execute_model` path and its count is READ by the multi-cache refusal
in `ModelRegistry::Forward`. Nothing CONSUMES the tables: no forward reads them,
because `ForwardQwen4ExpForConditionalGeneration` refuses by name and
`ModelRegistry::Forward` refuses every multi-cache topology. That is carried
under `## Owed` and named in the commit and pull-request bodies.

### The RED, before the change

The carrier landed first (the two `MultiKvCacheIndex` fields, the two accessors
and the refusal's new clause) with NO gather wired, so the red is behavioural
rather than a compile error — the accessor exists and answers `nullptr`:

```
tests/vllm/v1/worker/test_runner.cpp:2498: ERROR:
  CHECK( msg.find("block tables gathered for 3 of 3 published group(s)")
         != std::string::npos ) is NOT correct!
  values: CHECK( 18446744073709551615 != 18446744073709551615 )

tests/vllm/v1/worker/test_runner.cpp:2508: FATAL ERROR:
  REQUIRE( bt != nullptr ) is NOT correct!
  values: REQUIRE( nullptr != nullptr )
  logged: g := 0
```

### Why the fixture discriminates

`MakeQwen4ExpShapedKvConfig()` is this row's own miniature: `FullAttentionSpec` +
one uniform `MambaSpec` + the indexer `MLAAttentionSpec`. The case drives
`GPUModelRunner::execute_model` — the production entry point — with a 20-token
prompt over a 16-token block, so the sequence spans TWO blocks with a PARTIAL
final one, and gives each of the three groups a DISTINCT two-block list with no
fixed point: `{6, 2}`, `{4, 7}`, `{5, 3}`. An identity table would have proven
nothing, because a body that ignores the table, returns the logical indices, or
reads another group's table agrees with `{0, 1}`. The expectation is the literal
list handed to `add_row`, never a value read back out of the runner.

### The battery

Each mutation is a single edit to `src/vllm/v1/worker/gpu/runner.cpp`, applied to
a file whose pre-mutation sha256 is
`b1fcc71a36ee1ac02b87cbc8786958f301177661f3ce07287911d5785c3ae889`, built to a
recorded rc BEFORE any test was run, and restored to that same sha256.

| ID | Mutation | Build rc | Result |
|---|---|---|---|
| M1 | gather every group with the WRONG id — `gather_block_table(gdn_group_id_, …)` instead of `(g, …)` | 0 | **RED**, 6 of 44: groups 0 and 2 both read `{4, 7, 0, …}` where `{6, 2, …}` and `{5, 3, …}` are owed, and both inequality assertions fire |
| M2 | OFF-BY-ONE in the group index — `g == 0 ? 0 : g - 1` | 0 | **RED**, 4 of 44: group 1 reads group 0's `{6, 2}` and group 2 reads group 1's `{4, 7}` |
| M3 | REACHABILITY — the production call site `if (multi_cache_topology_) gather_group_block_tables(num_reqs);` in `execute_model` DELETED | 0 | **RED**, 2 of 1001 over the whole suite: the refusal reports no gather and every group answers `nullptr` |

M3 is NOT vacuous: there IS a production call site and deleting it reds the gate.
What it proves is bounded, and the bound is worth writing down — it proves the
RUNNER reaches the gather, not that any forward reaches the tables. No forward
does; see the paragraph above and `## Owed`.

### Counts, before and after, on the same tree

`tests/test_runner`, one binary, the two new cases excluded by name to get the
"before" figure rather than rebuilding a second tree:

| | test cases | assertions |
|---|---|---|
| before (`-tce` both new cases) | 32 | 990 |
| after | 34 | 1038 |

`tests/test_qwen4_exp_scaffold` is unchanged at 12 cases / 296 assertions, which
is the count its own `## Owed` paragraph in `qwen4_exp_registry.cpp` records.

### The refusal string was falsified by this change, and repaired in it — AND NEITHER SIDE OF THE MERGE WAS TRUE, TWICE

`ForwardQwen4ExpForConditionalGeneration`'s message enumerated "(2) reach for the
indexer side cache, whose group-2 block table `GPUModelRunner::gather_block_table`
never gathers (W5c-2)". That clause describes this commit's PARENT.

**Two waves landed on `main` between the battery above and the merge, and each
rewrote the same literal.** W5d-4 (`3f9177f7f`) removed the MoE clause; W5d-3
(`787373626`, #2276) removed the paged-QSA-consumer clause and left the group-2
clause standing. So at the final merge `main`'s literal enumerated ONE item — the
group-2 block table this change closes — and this branch's literal enumerated ONE
item — the paged QSA consumer W5d-3 closed. Each side was exactly one item too
long, and taking EITHER side verbatim would have put a false statement on `main`.
The literal is therefore resolved BY HAND, and the resolved count is the SET
DIFFERENCE of the five-item survey against every landed wave: **ZERO**. Read off
the tree rather than off either side's prose — item 1 is `vt::RmsNormGroup`, item
2 is `RunQwen4ExpQsaBlockPaged`, item 3 is
`GPUModelRunner::gather_group_block_tables`, item 4 is `qwen4_exp_moe.{h,cpp}`,
item 5 is `BuildMropeCosSinHost` declared in `qwen3_5_mrope.h`; all five resolve.
The enumeration is not DELETED, because a survey that falls silent reads as an
unfinished one: the message says the prerequisites are done and names what is
left, which is the LAYER LOOP itself (#2031) plus the QSA indexer side cache's
paged STORE, and it still refuses because #2031 is unwritten.

The emitted bytes were read back out of the running hook ON THE MERGED HEAD (a
temporary `MESSAGE` in the scaffold subcase, removed and the file restored to its
HEAD sha256
`32ee46d64b3045cbf852c387a2cb106bc46d391ed9fd19b4b6cbc013afac9b07`), not grepped
out of the source — `test_qwen4_exp_scaffold.cpp:767` pins substrings of that
string and therefore pins its PRESENCE, never its truth, and all five of its
substrings survive every wrong variant of this literal. What the hook emitted:

```
vt: Qwen4ExpForConditionalGeneration: the forward is not ported yet. The ops
and block seams ARE on main (W2/W3/W4/W6a/W5a/W5b-1..6, W5c-1, W5c-2, W5d-1,
W5d-2, W5d-3, W5d-4), and with W5c-2 ZERO of the five prerequisites #2249
surveyed remain: GPUModelRunner now gathers every published group's block table,
so the map into the QSA indexer side cache's pages reaches the forward. What is
missing is the LAYER LOOP itself, Qwen4ExpTextModel::Forward, owned by #2031,
with that side cache's PAGED STORE still owed beside it. ModelRegistry::Forward
additionally refuses any multi-cache topology by name, and this model publishes
one. See .agents/specs/qwen4-exp-flash-next.md and issues #2031 and #1978. at
.../src/vllm/model_executor/models/qwen4_exp_registry.cpp:266
```

The `vt: ` prefix and the trailing ` at <path>:266` are what `VT_CHECK` adds; the
line breaks above are presentational and the emitted string is one line, 794
characters as captured. The `:266` is the refusal's live line on this merged head
and is the anchor the `## Owed` bullet above cites. The suite printed
`test cases: 12 | 12 passed` and `assertions: 296 | 296 passed` on the same run,
so reading the bytes cost the gate nothing.

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

- **A REFUSAL THAT ENUMERATES PROSE GOES STALE SILENTLY, AND NOTHING PREVENTS THE
  FOURTH INSTANCE.** [#2288](https://github.com/mudler/vllm.cpp/issues/2288) is
  fixed by [#2265](https://github.com/mudler/vllm.cpp/pull/2265) for the two
  items that had gone false, but only those two: the DURABLE fix is not made
  here. `tests/vllm/models/test_qwen4_exp_scaffold.cpp:767` pins that five
  substrings are PRESENT, never that any of them is still TRUE, so a refusal
  listing finished work satisfies every assertion — it is a spelling gate, not a
  truth gate. This is the third instance on this row in one day (#2276 for the
  paged QSA consumer, #2254 for the opposite polarity, an understated refusal).
  A truth-linked check — each enumerated item naming a symbol whose absence the
  suite verifies — or a convention that the refusal enumerates ISSUE NUMBERS
  rather than prose would close it, and both are larger than this flow and belong
  to whoever owns the reachability convention. Recorded here so the residual has
  a named home rather than living only in the issue.

  **AND THE FOURTH INSTANCE THEN ARRIVED, EXACTLY HERE.** Merging `main` into
  W5d-4 ([#2249](https://github.com/mudler/vllm.cpp/issues/2249) item 4) made the
  refusal's third enumerated item — "an adapter from the stacked [E, I, H]
  qwen4_exp MoE tensors onto MoeBlockWeights" — false, because that adapter is
  what W5d-4 adds; landing it unedited would have shipped a commit whose own
  product output denies the commit, which is the #2288 finding word for word.
  It is repaired in this flow and the count further down is restated with it.
  The heading above is left standing rather than softened, because it was right:
  nothing mechanical caught this one either, a reading did, and the durable fix
  is still owed.
- **W5d-1 (#2249 item 1) lands UNREACHED, by AGENTS.md "Nothing lands dead".**
  `vt::RmsNormGroup` (`include/vt/ops.h`, dispatcher `src/vt/ops.cpp`, CPU kernel
  `RmsNormGroupKernel` in `src/vt/cpu/cpu_ops.cpp`, name in
  `src/vt/op_provider.cpp`) is reached at this merge commit only by
  `tests/vt/test_ops_rms_norm_group.cpp`. No production entry point calls it:
  `ModelRegistry::Forward` is the only one this architecture has, and
  `ForwardQwen4ExpForConditionalGeneration`
  (`src/vllm/model_executor/models/qwen4_exp_registry.cpp`) still refuses by name
  before any downcast, so the PLE block that will hold the three
  `Qwen4ExpTextRMSNorm(group_size=hidden_size)` calls does not exist to call it
  from. Wiring it is owned by row `MODEL-MM-QWEN4-EXP` under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031), tracked by campaign
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978), and gated by
  [#2249](https://github.com/mudler/vllm.cpp/issues/2249). The W2 host reference
  `qwen4_exp_ple.cpp` was NOT rerouted through the op: its `GroupedRmsNorm`
  accumulates in double by deliberate choice ("a reference choice and not a
  divergence"), it is file-local and itself unreached, so routing it would have
  changed a golden-gated number and bought no reach. **M5** is the load-bearing
  proof at the layer that does exist: registering the kernel on
  `DeviceType::kCUDA` instead of `kCPU` leaves the build at rc 0 and reds the
  suite BY REFUSAL, `vt: no kernel for op RmsNormGroup (id 140) on device cpu
  (type 0)`, so the dispatcher path is live rather than vestigial and the
  `op_provider.cpp` name entry is live with it. **M4 is a different reading and
  is recorded as one**: deleting the `RegisterOp(OpId::kRmsNormGroup, ...)` line
  outright is a BUILD refusal, `error: 'RmsNormGroupKernel' defined but not used
  [-Werror=unused-function]`, because that registration is the only reference to
  a kernel defined in the anonymous namespace `src/vt/cpu/cpu_ops.cpp` opens at
  :24. No suite runs at all under M4, so it cannot red one; what it proves is the
  link, which is why M5 exists beside it. The battery table two sections above
  states both correctly; this sentence did not, and a mutation misread as a test
  verdict is exactly the confusion the battery was run to prevent.
- **The CUDA arm of `vt::RmsNormGroup`.** Not written, for the reason W5b-3 and
  W5b-4 give for theirs: it could not be gated on this CPU-only host with no
  lease, and an ungated kernel is worse than an absent one. Nothing registers for
  any device but `kCPU`, so the dispatcher refuses BY NAME rather than falling
  back — which is the whole argument for a separate OpId over a `group_size`
  field on `RmsNormArgs`, since a new field on that shared struct would be
  silently ignored by the backends that already register `kRmsNorm` and read only
  `eps` and `gemma` off it. **SIX register it in total** — `kCPU`
  (`cpu_ops.cpp:3750`), `kCUDA` (`cuda_ops.cu:3917`), `kROCM`
  (`rocm_ops.hip:118`), `kVULKAN` (`vulkan_ops.cpp:1626`), `kMETAL`
  (`metal_ops.mm:1108`) and `kTENSTORRENT` (`tenstorrent_ops.cpp:5323`) — which
  is FIVE besides the `kCPU` this wave teaches, and five others is the number
  that carries the argument, because they are the ones that would answer a
  grouped request with a whole-row norm.

  This entry said "four", which is wrong under either reading, and the
  correction rides here because this is the paragraph the W5d-1 review already
  sent back. `include/vt/ops.h:634` says "five backends" and is NOT corrected:
  its next clause is "so a CUDA or Metal caller would get a whole-row norm
  back", so it is counting the five OTHER backends and is consistent with this
  enumeration. The two records are reconciled here rather than left to read as
  a contradiction, and no product file is touched to do it.

  The arm owes one decision this wave did not make for it: whether the per-group
  sum of squares reduces in f32 (as the CPU arm does, mirroring `x.float()` at
  `modeling_qwen4_exp.py:174` and `RmsNormKernel` beside it) or in a wider
  accumulator once the group is 2560 wide rather than 6.
- **W5d-2 (#2249 item 5): the mRoPE seam is REACHED, but only by a caller that
  is not itself routed from a production entry point.** `BuildMropeCosSinHost`
  now has external linkage behind
  `include/vllm/model_executor/models/qwen3_5_mrope.h`, and `qwen3_5.cpp`'s two
  production call sites resolve through that declaration — deleting them reds
  `test_qwen3_5_moe_vision`'s
  `qwen3_5_moe_vl_image_forward_uses_MRoPE_positions_not_plain_1d`. The hop above
  is the gap: `Qwen3_5VLGenerateGreedy`, `Qwen3_5VLGenerateGreedyVideo`,
  `Qwen3_5MoeVLGenerateGreedy` and `Qwen3_5MoeVLGenerateGreedyVideo` are DEFINED
  in `src/vllm/model_executor/models/qwen3_5.cpp:9892,9915,9960,9974` and declared
  in `qwen3_5.h` / `qwen3_5_dense.h`, and a tree-wide grep over
  `src/ include/ examples/ tools/ benchmarks/` finds no other occurrence — every
  CALLER is in `tests/`. The registered factories for
  `Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration`
  carry no multimodal hook, so `ModelRegistry::Forward` cannot arrive; the tree
  says so for the sibling driver at
  `include/vllm/entrypoints/openai/chat_mm.h:266-267`. This condition PREDATES
  the extraction and the extraction does not change it in either direction, but
  it is named here because `## Nothing lands dead` asks the question at every
  merge. TWO owners, because there are two ways to close it, and
  `.agents/reachability.md` asks for a row ID and an issue for each rather than
  a description:

  1. **The qwen4_exp call.** The qwen4_exp layer loop will call this seam from a
     path that IS routed through `ModelRegistry::Forward`. Row
     `MODEL-MM-QWEN4-EXP`, W5b under
     [#2031](https://github.com/mudler/vllm.cpp/issues/2031); the extraction
     itself is this row's and is tracked by
     [#2249](https://github.com/mudler/vllm.cpp/issues/2249).
  2. **Request routing to the VL drivers.** Getting an image or video request
     from the registered forward to `Qwen3_5VLGenerateGreedy` and its three
     siblings is an ENGINE seam and not this model port's. Row
     **`ENG-MM-QWEN36-VL-FORWARD`** (`.agents/engine-matrix.md`, state
     `ACTIVE`), which already owns `BuildMropeCosSinHost`, the shared
     `VLGenerateCoreGdn` and the two Qwen3.6-27B dense drivers; the two MoE
     drivers additionally sit under row
     `MODEL-MM-qwen3-5-qwen3-5-moe-for-conditional-generation` and
     [#891](https://github.com/mudler/vllm.cpp/issues/891). Tracked by
     [#2257](https://github.com/mudler/vllm.cpp/issues/2257), filed while
     landing this wave because nothing tracked it before: the gap is real, it
     predates the extraction, and it had no issue of its own. An earlier
     revision of this entry named this owner only as "the mm-forward row",
     which is a description and not a record.
- **W5d-4 ([#2249](https://github.com/mudler/vllm.cpp/issues/2249) item 4) lands
  UNREACHED, by AGENTS.md "Nothing lands dead".**
  `src/vllm/model_executor/models/qwen4_exp_moe.{h,cpp}` —
  `Qwen4ExpMoeBlockWeights`, `Qwen4ExpMoeHfConfig` and `RunQwen4ExpMoeBlock` — is
  reached at this merge commit ONLY by
  `tests/vllm/models/test_qwen4_exp_moe.cpp`. No production entry point calls
  it, and the MECHANISM is not the one an earlier draft of this bullet named.
  `ForwardQwen4ExpForConditionalGeneration` EXISTS
  (`src/vllm/model_executor/models/qwen4_exp_registry.cpp:142`) and IS registered
  as the model's `.forward` hook (`:494`); its entire body is one
  `VT_CHECK(false, ...)` refusal-by-name (`:266`), placed ahead of any downcast.
  So `ModelRegistry::Forward` reaches a real hook and that hook refuses
  `Qwen4ExpForConditionalGeneration` by name before a layer runs. The conclusion
  — nothing production-side reaches this adapter — is unchanged; "the function
  does not exist" was wrong and is corrected here rather than left to be
  reasoned from. The wiring is owned
  by row `MODEL-MM-QWEN4-EXP` and tracked by
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031) (the layer loop) under
  campaign issue [#1978](https://github.com/mudler/vllm.cpp/issues/1978). The
  reachability mutation is therefore vacuous here and is recorded as such rather
  than reported as a pass: deleting a production call site cannot red a gate when
  there is no production call site. What the suite DOES prove is that it enters
  through the adapter — deleting the adapter's keep-quant wiring reds it
  (mutation M4).
  Six further things W5d-4 owes:
  - **`norm_topk_prob` is not representable through this seam.** `MoeBlock`
    hardcodes `renormalize = true` (`qwen3_5.cpp:7242`) and `HfConfig` carries no
    such field, so a config that turned it off could not be honoured and the
    adapter cannot refuse what it cannot see. Upstream's default is `True`
    (`configuration_qwen4_exp.py:163`) and the released checkpoint does not
    override it, so nothing is wrong today; `Qwen4ExpParams` still does not carry
    the field, which this section already owed above.
  - **The bf16 arm is ineligible for the CUDA fast grouped-bf16 MoE.**
    `MoeBf16FastLayoutOk` (`qwen3_5.cpp:842-863`) requires per-expert `[H, I]`
    with `nk == false`, and the adapter's zero-copy views are the tower's own
    `[I, H]` with `nk == true`. It therefore falls through to the reference
    per-expert loop on CUDA, exactly as the 35B MTP producer already does. The
    alternative is a transposing copy, which is 240 GB across the stack at the
    released geometry, so this is a deliberate trade and not an oversight — but a
    grouped bf16 path that reads the tower orientation is owed if a bf16 arm ever
    becomes the shipped one. It is not one today: all seven staged checkpoints are
    quantized.
  - **The NVFP4 expert arm is refused by absence, not by name.**
    `Qwen4ExpMoeWeights` has no `Nvfp4Weight` fields and `LoadStackedExperts` has
    no fp4 branch, so `MoeBlockWeights::expert_*_fp4` are left empty and the
    seam's `fp4` predicate is false. Separately, and NOT this wave's to fix,
    filed as [#2275](https://github.com/mudler/vllm.cpp/issues/2275) and owned by
    this row:
    `LoadStackedExperts` (`qwen4_exp_weights.cpp:148-167`) handles only
    `kKeepQuant` and falls through to `ExpandBf16` for BOTH `kKeepF16` and
    `kNvfp4Fp4`, which `GgufLoadPolicy::Route` can return for
    `kStackedExpertWeight` (`gguf_keep_quant.cpp:59-60`). That silently produces a
    residency the policy did not ask for; at the released geometry it is a 240 GB
    allocation rather than a wrong answer, so it fails loudly, but it belongs to
    the W5a loader and is recorded here so the next reader does not read the
    adapter's two arms as the loader's full range.
  - **The routed top-k weights reach the experts f32, which is WIDER than the
    oracle, and the width is owed rather than defended.** Upstream casts the
    renormalized top-k weights back to the model dtype
    (`router_top_value.to(router_logits.dtype)`, `modeling_qwen4_exp.py:914`);
    our shared seam keeps them f32, because `vt::MoeRouterTopK` writes an f32
    `dtw` (`qwen3_5.cpp:7239-7242`) and every Qwen MoE in this tree reads that
    field. An earlier draft of the suite header called the seam "the more
    precise of the two". That is exactly the argument AGENTS.md §"Inherit vLLM
    defaults" exists to refuse: a token gate cannot see a dtype that is too
    wide, so "more precise" is never a reason to be wider than the oracle. The
    honest statement is that the adapter INHERITS the width from the seam, has
    no way to narrow it without diverging from every other Qwen MoE here, and
    that narrowing `dtw` to the model dtype is a seam-level change owed to the
    shared seam rather than to this adapter. Nothing measures it today: at
    top_k = 3 it is one bf16 rounding of a value in [0, 1] per pair, below this
    suite's tolerances, so it needs a gate of its own.
  - **The keep-quant arm is value-proven at Q8_0 ONLY, and against a fixture
    whose bf16 towers are `nk = false` where the loader's are `nk = true`.**
    `tests/vllm/models/test_qwen4_exp_moe.cpp` builds its keep-quant towers as
    hand-written Q8_0 blocks. The adapter is dtype-generic on that arm by
    construction — it re-declares the tower rank 2 and copies no bytes, and
    `vt::MatmulBTQuantGrouped` accepts any `IsBlockQuant` dtype — but NO k-quant
    tower is executed through it here, and every released Qwen4-Exp checkpoint
    is a k-quant (Q2_K..Q6_K, IQ1_*). A k-quant value case is owed, and it is
    owed at the loader rather than at the adapter, because it needs
    `OwnGgufQuantBlocks` output rather than a hand-built block. Separately, the
    fixture's SOURCE bf16 towers are built `nk = false` where
    `LoadStackedExperts` produces `ExpandBf16(..., /*nk=*/true)`: the adapter
    stamps `nk = true` on the per-expert views it hands the seam either way, so
    the gated bytes and the gated orientation are the production ones, but the
    fixture is not the loader's own output and this section says so rather than
    letting a reader infer that it is.
  - **The keep-quant tolerance is MEASURED, not derived.** An earlier draft of
    the suite header justified it by claiming the gate/up activation is exactly
    q8_0-representable, "amax is exactly 127*2^-8 per row". `QuantizeRowQ8_0`
    (`src/vt/cpu/cpu_quant_act.cpp:52-81`, its per-block `amax` loop at
    `:58-69`) computes `amax` per 32-ELEMENT BLOCK,
    not per row, and the fixture's `HiddenCodes()` forces `|code| = 127` at
    element 0 of each row only. `kH = 64` is TWO blocks, so block 1 takes an
    arbitrary `amax`, its `d` is not `2^-8`, and the gate and up projections
    carry quantization error as well as the down projection. The bound the suite
    asserts is therefore what it measures, not what it derives; it is printed by
    a `MESSAGE` on every run, and it is an order of magnitude below every
    mutation margin. Forcing `|code| = 127` in every block of every row would
    restore the derivation, and it is deliberately NOT done here: it moves the
    router logits, hence the routing, hence the seven mutation margins an
    independent review has already reproduced against this fixture. The
    derivation is owed to whichever change next has a reason to move the
    fixture.

- **W5b-4 (#2167) lands UNREACHED, by AGENTS.md "Nothing lands dead".**
  `vt::Qwen4ExpQsaCompress` and `vt::Qwen4ExpQsaGatherAttention`
  (`include/vt/ops.h`, dispatchers `src/vt/ops.cpp`, CPU kernels
  `src/vt/cpu/cpu_qwen4_exp_qsa.cpp`) are reached at this merge commit only by
  `tests/vllm/models/test_qwen4_exp_qsa_device.cpp`. No production entry point
  calls either: `ModelRegistry::Forward` is the only one this architecture has,
  it is all-or-nothing, and `ForwardQwen4ExpForConditionalGeneration`
  (`src/vllm/model_executor/models/qwen4_exp_registry.cpp`) still refuses by name
  before any downcast. The wiring is owned by row `MODEL-MM-QWEN4-EXP` and by W5b
  under [#2031](https://github.com/mudler/vllm.cpp/issues/2031), tracked by
  campaign [#1978](https://github.com/mudler/vllm.cpp/issues/1978), and reaching
  the ops from the runner's caches additionally waits on
  [#2131](https://github.com/mudler/vllm.cpp/issues/2131). Mutations M24 and M25
  are the load-bearing proof at this layer: deleting either `RegisterOp` line
  reds the suite, so the dispatcher path is live rather than vestigial.
- **The CUDA arm of both QSA ops.** Not written, because it could not be gated on
  this CPU-only host with no lease, and an ungated kernel is worse than an absent
  one. Nothing registers for any device but `kCPU`, so the dispatcher refuses by
  name rather than falling back. That arm owes three decisions this wave did not
  make for it: the reduction width for the pooled key's sum of squares (`f32`
  here, in the host reference's order, because that is the order the goldens were
  dumped in); a DEVICE-side `keys_visited` counter and its copy-back, since
  `Qwen4ExpQsaAttnArgs::keys_visited` is a host pointer and cannot survive a
  launch; and whether the gather is a genuine address-generated gather on the
  device or degrades to a mask, which is the whole point of the row and is
  exactly what a CPU host cannot measure.
- ~~**W5b OWES THE INDEXER COMPOSITION IN PRODUCTION CODE, AND FOUR SETTINGS WITH
  IT.**~~ **DISCHARGED by W5b-5 ([#2211](https://github.com/mudler/vllm.cpp/issues/2211)),
  and the VALUE gate it demanded exists.** `Qwen4ExpQsaIndex`
  (`src/vllm/model_executor/models/qwen4_exp_qsa_block.cpp`) composes the three
  ops and states all four settings at one site, each beside the upstream line it
  mirrors and the mutation that reds it. The gate is
  `tests/vllm/models/test_qwen4_exp_qsa_block.cpp`, against the oracle's OWN
  pre-top-k `scores` tensor: fed `k...IdxQPost` and `k...IdxKRaw` — the oracle's
  own roped query and raw keys — the composed logits are BIT-IDENTICAL to it,
  max abs **0** over scales of 3.37 and 6.24 across 12 and 60 logits. M1
  (inherit `n_head ** -0.5`) and M2 (drop the softmax scale) both RED on that
  case, which is the repair for M26's recorded survival. The paragraph below is
  kept because it is the reasoning the repair rests on, not because the debt is
  open. The composition WAS in exactly one place, the `RunIndexer` helper in
  `tests/vllm/models/test_qwen4_exp_qsa_device.cpp`, and nothing under `src/`
  enforced any of the four settings the collapse depends on:
  1. `weights` is all ones (`[T, index_n_heads]`), which is what collapses the
     per-head fold to a single constant. M27 is its red control.
  2. `n_head_scale == 1.0f`, NOT DeepSeek-V4's `n_head ** -0.5`, which QSA has no
     tensor for.
  3. `softmax_scale == index_head_dim ** -0.5`, QSA's own scale.
  4. `win_end == kv_len / compress_ratio` per query token — the COMPLETE visible
     blocks, not the whole cache. M28 is its red control.
  Two of the four have no gate that would catch a wrong value in production:
  M26 records that `n_head_scale` is invisible to selection BY CONSTRUCTION,
  because top-k is invariant under a positive rescale of every score, and
  `softmax_scale` is invariant for the same reason. That is why the W5b-5 gate is
  a VALUE gate on the logits and not a selection gate.
- **A single-pass online softmax for the gather.** The CPU kernel makes two
  passes over the selected rows per query head, which is why the honest read
  count is `selected * num_q_heads * 2`. A single-pass rewrite legitimately
  halves it, and `kReadsPerRowPerHead` in the device suite is where that constant
  gets re-derived on purpose rather than silently absorbed. No speed claim is
  admissible from this row until G2 passes, so this is owed, not deferred work.
- **The `bf16` operand arms of both ops are declared and UNGATED.** The
  dispatchers accept `f32`/`bf16` and the kernels widen through `LoadF32At`, but
  every fixture is `f32`-valued and the goldens are `f32` arrays of
  bf16-representable numbers, so no case stores a bf16 tensor. That is honest
  rather than complete: the `round_intermediates_to_bf16` flag is gated (M7, M9),
  the bf16 STORAGE path is not.
- **`Qwen4ExpQsaCompress` assumes a CONTIGUOUS visible prefix**, as the W4 host
  reference does. Upstream forms blocks over `local_visible_indices` of a padded
  batch; a serving engine's ragged batch has no interior masking, so the two
  coincide and block `b` is exactly tokens `[CR*b, CR*b + CR)`. The op REFUSES a
  key count that is not a whole number of complete blocks (M20 reds that refusal)
  but it cannot detect an arbitrary visibility set, and nothing yet does.
- **The side cache's paged store.** `QsaSideCacheSpec` (W4) says what the cache
  costs and `QsaCompressedSlot` says which slot a token writes; this op writes a
  DENSE `[num_blocks, head_dim]` array and not a paged one. The block-table store
  belongs to the wave that gives QSA a real KV-cache group, which is blocked on
  [#2131](https://github.com/mudler/vllm.cpp/issues/2131). **This is a PRODUCTION
  obligation only.** The wave's first draft also recorded the fetch-level PROOF —
  a cache whose unselected blocks fault when touched — as waiting on the same
  store. It never was: the fresh review built it out of `mmap` and
  `mprotect(PROT_NONE)` inside the test, it is the case `the gather never FETCHES
  an unmapped unselected row`, and M11c is the paired control showing it convicts
  a body the NaN poison cannot see. Nothing about the instrument is owed.

- **W5b-5 ([#2211](https://github.com/mudler/vllm.cpp/issues/2211)) lands
  UNREACHED, by AGENTS.md "Nothing lands dead".**
  `src/vllm/model_executor/models/qwen4_exp_qsa_block.{h,cpp}` — `Qwen4ExpQsaIndex`
  and `RunQwen4ExpQsaBlock` — are reached at this merge commit only by
  `tests/vllm/models/test_qwen4_exp_qsa_block.cpp`. No production entry point
  calls either: this architecture's only one is `ModelRegistry::Forward`, it is
  all-or-nothing, and `ForwardQwen4ExpForConditionalGeneration`
  (`src/vllm/model_executor/models/qwen4_exp_registry.cpp`) still refuses by name
  because the LAYER LOOP is not written. The wiring is owned by row
  `MODEL-MM-QWEN4-EXP` under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031), tracked by campaign
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978). Also owed from that
  wave, each named rather than discovered later:
  - **THE LAYER LOOP, WHICH IS NOW THE ONLY THING BETWEEN THIS ROW AND A TOKEN.**
    Every seam and every op the forward needs is on `main` — `RunGdnBlockPaged`
    (W5b-1), the two gated-residual ops (W5b-2), `vt::Qwen4ExpPleConv` (W5b-3),
    the two QSA ops (W5b-4) and now the QSA BLOCK (W5b-5). What has no production
    shape yet is: the PLE block (the n-gram hash and its gather composed with
    `vt::Qwen4ExpPleConv`), the GDN and MoE weight adapters
    (`Qwen4ExpGdnWeights` -> `GdnLayerWeights`, `Qwen4ExpMoeWeights` ->
    `MoeBlockWeights`; the GGUF loader mirrors `qwen3_5_gguf_weights.cpp` name for
    name and shape for shape, so the adapters are field aliasing rather than
    arithmetic), the 10240-wide hyper-connection stream through the per-layer
    loop, the interleaved-mRoPE cos/sin table build, and the terminal mixer plus
    `lm_head`.
  - **THE `hc_norm` POLARITY IS A TRAP THE LAYER LOOP WALKS INTO, and it is
    recorded here because W5b-5 hit the same shape and got it right by accident
    of scope.** `LoadNormBf16(..., unshift=true)` INVERTS the converter's baked
    `+1`, so every gamma the loader stores is the RAW HuggingFace value, centred
    on 0. `vt::Qwen4ExpQsaCompress` wants exactly that and applies `(1 + w)`
    itself; `vt::RmsNorm` wants it under `RmsNormArgs::gemma = true`, which is
    what this block passes and what mutations M9/M10/M11 red. But
    `vt::Qwen4ExpGatedResidual` documents the OPPOSITE convention — "hc_norm_w is
    vLLM's parameterization, i.e. ALREADY `1 + w_hf` … This op never adds 1" —
    so the layer loop must fold `hc_norm`, `norm_key`, `norm_query` and
    `norm_conv` with `vllm::qwen4_exp::HcNormWeightFromHf` before handing them to
    that op. Miss it and every gated residual applies a near-zero scale, which
    reads as a checkpoint bug rather than a port bug. Nothing gates this today,
    because the layer loop is the first caller. **Tracked as
    [#2218](https://github.com/mudler/vllm.cpp/issues/2218)**, which is its own
    wave and deliberately not repaired here. The contradiction is visible AT THE
    LOAD SITE and does not need the op to be read to be seen:
    `qwen4_exp_weights.cpp:258-263` argues FOR the fold in its own comment — "the
    fold is what makes the file's value the multiplier our own `out * weight`
    grouped norm wants", corroborated elementwise on three published artifacts —
    immediately above the line that strips it with `unshift=true`.
  - ~~**The PAGED cache.** This block takes CONTIGUOUS per-sequence K/V and a
    [#2131](https://github.com/mudler/vllm.cpp/issues/2131) and on W5c.~~
    **HALF DISCHARGED by W5d-3 ([#2249](https://github.com/mudler/vllm.cpp/issues/2249)
    item 2)**, and the half is named so nobody reads this as done. The QSA
    layers' K/V — KV group 0, the `FullAttentionSpec` — now has a paged consumer:
    `Qwen4ExpQsaPagedCaches` + `RunQwen4ExpQsaBlockPaged`, over a
    `kv_block_table`/`kv_block_size` ADDRESS MODE inside the same
    `vt::Qwen4ExpQsaGatherAttention` rather than a second op. The INDEXER side
    cache is still contiguous: that is KV group 2, the `MLAAttentionSpec`, and its
    paged store is the separate entry above and #2249 item 3 (owed as W5c-2). What
    W5d-3 did NOT need from #2131 is worth recording, because this bullet asserted
    the dependency for three waves: the K/V paged read needs only a block table and
    a slot mapping, both of which the runner already builds for every full-attention
    model, and none of the multi-state recurrent work #2131 owns.
  - **The RAGGED-BATCH form.** `kv_lens[t] = past_len + t + 1` is built inside the
    block from a CONTIGUOUS visible prefix. Upstream's general form reads an
    arbitrary visibility set out of a padded batch's mask, and the ops' own
    `## Owed` already records that nothing here can detect one; the block
    inherits that limit rather than adding to it.
  - **The cos/sin table BUILD**, and with it the interleaved-mRoPE section
    layout. The block takes the tables as operands in BOTH layouts the two ops
    want — a bf16 PACKED `[P, rot]` cos|sin cache for `vt::RopeFromCache` and two
    f32 FULL `[P, rot]` tables for `vt::Qwen4ExpQsaCompress` — and asserts each by
    name. It also CROSS-CHECKS the two against each other, which it did not when
    the header first claimed it did: equal heights, then a BOUNDED SAMPLE of rows
    (row 1, the midpoint and the last row) compared value for value at one bf16
    ulp. Row 0 is not a probe, because cos is 1 and sin is 0 at every frequency
    there and it agrees under every construction difference. What the sample
    cannot see is a single corrupted row; what it does see is every table-wide
    difference a layer loop can make — a different theta, a different
    `rotary_dim`, an interleaved pack, swapped halves, an off-by-one position
    offset, or a position scaling applied to one table and not the other. A FULL
    comparison is deliberately not paid: it would be O(P * rot) per QSA layer per
    step to re-check a constant. The wave that builds them still owes the case
    with three DISTINCT position streams this spec already records as unowned.
  - **The bf16 STORAGE arm is the only arm.** The block refuses an f32 `hidden`
    by name, and the reason is a shared-surface fact rather than a preference:
    every `vt::` output-gate op in this tree — `SigmoidGateBf16`,
    `SharedExpertGate` — stores bf16 on every backend, because vLLM resolves one
    model dtype and this tree inherits that polarity. An f32 arm would have to
    widen a dispatcher across five backends this host cannot gate, and the
    refusal says so.
  - **The CUDA arm**, which is the QSA ops' own owed item and not a new one. The
    block adds no arithmetic, so it inherits that debt unchanged. TWO things ARE
    new at the BLOCK level, and they are named here rather than folded into that
    inherited debt, because a device arm has to answer both and neither is
    visible from the ops:
    - **The indexer's per-call INDEX BUILD is done on the host.**
      `Qwen4ExpQsaIndex` materialises `ones` `[T, H]`, `win_start` `[T]` and
      `win_end` `[T]` into host vectors and hands each to a `DBuf`, and
      `RunQwen4ExpQsaBlock` does the same for `kv_lens` `[T]` — FOUR small
      host-to-device copies per QSA layer per step on a device queue. It also
      `VT_CHECK`s that `kv_lens` is CPU-resident and reads it on the host to build
      the window, which is a refusal a device-resident batch would hit by name.
      Upstream rebuilds exactly the same metadata on every call —
      `local_visible_indices` out of `torch.nonzero` on the mask row,
      `block_token_indices`, `group_starts` and `selected_token_indices`, inside a
      `for batch_idx / for query_idx` Python loop
      (`modeling_qwen4_exp.py:667-702`) — so the PER-CALL REBUILD is a faithful
      mirror rather than a divergence. What is not inherited is the transfer:
      upstream has one device and no H2D edge to pay, so the device arm owes the
      decision of where these four are built, and the nested Python loop is a
      reminder that the oracle is a reference implementation and not a
      performance model.
    - **The pooled BLOCK KEYS are recomputed over the ENTIRE cache every step.**
      The `block_keys` scratch is allocated per call and dropped, and
      `vt::Qwen4ExpQsaCompress` runs over cache rows `[0, complete_keys)` — O(kv)
      per layer per token, for a quantity that only ever GROWS by one block every
      `compress_ratio` tokens. Upstream does the same, and worse: it recomputes
      `pooled_keys` and `block_key_states` inside the per-query-token loop
      (`:679-686`), so its cost is O(kv) per query token per layer. So this is a
      faithful mirror of a reference implementation, and it is the shape that
      makes the incremental store worth having. The wave that gives QSA a real
      KV-cache group turns this scratch into the side cache's paged store and
      inherits the choice of whether to keep it incremental.
  - **The FETCH-level proof is inherited, not re-built.** The `mprotect(PROT_NONE)`
    unmapped-tail probe lives one layer down in
    `test_qwen4_exp_qsa_device.cpp`, and it is load-bearing for this block because
    the block's ONLY consumer call is `vt::Qwen4ExpQsaGatherAttention`. Mutation
    M20 is what says that call is the one under test: handing the consumer every
    VISIBLE block — a dense walk wearing a gather's clothes — reds 3 of 8 cases
    and 130 assertions.

- **W5d-3 ([#2249](https://github.com/mudler/vllm.cpp/issues/2249) item 2) lands
  UNREACHED, by AGENTS.md "Nothing lands dead".** The paged QSA consumer —
  `Qwen4ExpQsaPagedCaches` and `RunQwen4ExpQsaBlockPaged`
  (`src/vllm/model_executor/models/qwen4_exp_qsa_block.{h,cpp}`) together with the
  `kv_block_table`/`kv_block_size` address mode on
  `vt::Qwen4ExpQsaGatherAttention` — is reached at this merge commit only by
  `tests/vllm/models/test_qwen4_exp_qsa_block.cpp`. The reason is unchanged from
  W5b-5 and is not a property of this wave: this architecture's only production
  entry point is `ModelRegistry::Forward`, it is all-or-nothing, and
  `ForwardQwen4ExpForConditionalGeneration`
  (`src/vllm/model_executor/models/qwen4_exp_registry.cpp`) still refuses by name
  because the LAYER LOOP is not written. The wiring is owned by row
  `MODEL-MM-QWEN4-EXP` under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031), tracked by campaign
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978). The reachability
  mutation `.agents/reachability.md` prescribes has no site here for the same
  reason it had none for W5b-5: there is no production call site to delete.
  Also owed from this wave:
  - **The INDEXER side cache is still contiguous.** #2249 item 3 — KV group 2 is
    never gathered — is owed as W5c-2 and is deliberately not smuggled into this
    wave. `Qwen4ExpQsaPagedCaches::index_key` is `[max_kv, indexer_head_dim]`, so
    a forward built on this arm still needs a contiguous side cache per sequence.
  - **ONE REQUEST PER CALL.** `kv_block_table` is `[1, max_pages]` and the block
    refuses anything else by name. A ragged multi-request batch needs the
    `query_start_loc` plumbing `vt::PagedAttention` carries and this block does
    not, on top of the RAGGED-BATCH `kv_lens` item already owed above.
  - **An fp8 paged KV cache is REFUSED BY NAME.** `vt::Qwen4ExpQsaGatherAttention`
    has no dequantising read and no `k_scale`/`v_scale`, so an fp8 page would be
    read as floats — wrong tokens, not a crash, which is the exact failure
    `kv_cache_route.h` exists to prevent. The refusal is gated.
  - **The CUDA arm of the paged address mode**, inherited from the QSA ops' own
    owed CUDA arm and not a new debt: the address resolution is four lines of
    integer arithmetic in the same kernel body, so whatever answers the ops
    answers this.

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
- **W5b-3 ([#2156](https://github.com/mudler/vllm.cpp/issues/2156)) lands
  UNREACHED, by AGENTS.md "Nothing lands dead".** `vt::Qwen4ExpPleConv` and
  `src/vt/cpu/cpu_qwen4_exp_ple.cpp` are reached only from
  `tests/vllm/models/test_qwen4_exp_ple_device.cpp`. No production entry point
  calls them: the architecture's only one is `ModelRegistry::Forward`, which is
  all-or-nothing, and it has no `qwen4_exp` arm. The wiring is owned by row
  `MODEL-MM-QWEN4-EXP`, tracked by
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031) (the forward) under
  campaign issue [#1978](https://github.com/mudler/vllm.cpp/issues/1978).
  Also owed from that wave:
  - **The CUDA arm.** Not written, because it cannot be gated on a CPU-only
    host, and an ungated kernel is worse than an absent one. It inherits one
    decision: this CPU kernel accumulates its four taps in `double` and the
    device gate asserts BIT-IDENTITY with the W2 host reference at the model's
    10240-channel width. An f32-accumulating CUDA kernel does not inherit that
    identity — mutation M9 measures exactly this — so it must either accumulate
    wider or be gated against the pinned oracle directly.
  - **A bf16 `conv_state`.** The dispatcher refuses one by name.
    `CausalConv1dSpecUpdate` admits bf16 state on CUDA because a CUDA kernel
    there writes it; here nothing does, and admitting a dtype no arm can produce
    would be a promise with no kernel behind it. It is owed with the CUDA arm.
  - **Reaching the op from the runner's recurrent cache.** The op takes its state
    as an explicit `[N, C, (K-1)*dilation]` operand plus a per-sequence row
    index. That parameter is called `conv_state_indices`, after
    `CausalConv1dUpdate`'s parameter for the same axis, and it is deliberately
    NOT spelled `state_idx`, because upstream's `state_idx` selects
    one of a PLE layer's three states, and those three cannot be planes of one
    tensor because `cache_utils.py` keeps `conv_states` as a list with a
    per-entry `conv_kernel_size[state_idx]` and the widths are 4, 9 and 2, over
    different channel counts and, for the third, over integers. Resolving that
    selector is the caller's job, and the caller cannot exist until
    [#2131](https://github.com/mudler/vllm.cpp/issues/2131) generalises the
    runner's one-group/two-shape `MambaSpec`.
  - **The prefix-caching decision** for a conv state written by a chunked prefill
    shorter than nine columns, which `## Design` records as AMBIGUOUS and not
    resolvable from upstream, is untouched by this wave. This op reproduces
    upstream's zero-pad exactly; it does not decide what a cache HIT should
    restore.
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
  `0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a`. Those digests were
  the Hub API's `lfs.oid` values; a LOCAL sha256 was owed and is now recorded. All three
  were recomputed on the staged copy on 29 August 2026 and agree three for three, in
  [the ladder-arm evidence file](../../docs/bench-evidence/qwen4exp-llamacpp-ladder-arm-20260829.md). The "1224 tensors" count remains UNVERIFIED: shard 1 is the
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
- **W5b-1 (#2110) lands UNREACHED, by AGENTS.md "Nothing lands dead".**
  `RunGdnBlockPaged` and `BuildGdnStepInputs`
  (`include/vllm/model_executor/models/qwen3_5_gdn_block.h`, implemented beside
  `RunMoeBlock` in `src/vllm/model_executor/models/qwen3_5.cpp`) expose the
  qwen3_5 Gated DeltaNet block cross-TU so the `qwen4_exp` forward runs the SAME
  block its 36 `kLinearAttention` layers are, instead of growing a second copy
  of it. No production entry point calls either one at that commit: the
  Qwen3.5/3.6 forward keeps calling the anonymous-namespace `GdnBlockPaged`
  directly with its own `Dev` and its own step inputs, which is what keeps that
  path byte-identical, and the only caller of the public pair is the seam case in
  `tests/vllm/models/test_qwen3_5_gdn_spec_routing.cpp`. Row
  `MODEL-MM-QWEN4-EXP` owns the wiring, and W5b —
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031), under
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978) — is the wave that
  composes it into `Qwen4ExpTextModel::Forward`. `RunMoeBlock` landed the same
  way and for the same architecture. This entry also discharges the FIRST
  sub-bullet of the W5b entry below for two of the seven names it lists: after
  #2110, `GdnBlockPaged` and (since `f730eb11c`) `MoeBlock` are both reachable
  from another translation unit. `FullAttnBlockPaged` and `RunLayerPaged` stay
  sealed and are not needed — this architecture has no full-attention layer and
  its own layer shape — and `StepDevInputs` / `BuildStepDevInputs` stay sealed on
  purpose, reached through the opaque `GdnStepInputs` handle.
- **A NUMERIC gate on the seam's `dh_fp8` argument needs a CUDA host.** The fp8
  W8A8 GDN input-projection tower is CUDA-only — `MatmulFp8CutlassD` and
  `MatmulFp8CutlassPreQuantD` both refuse unless `kMatmulFp8CublasLt` is
  registered — so on CPU the argument is gated by WHICH of those two leaves
  refuses, in both directions (`## Mutation record — W5b-1`). That is a genuine
  discriminator and it closes mutation C, but it compares no number. On a device
  that supports fp8 the same sub-case can compare the forwarded arm's output
  against `QuantFp8Static(h, input_scale)` fed through the plain arm, which is
  what the two leaves are documented to make identical. Owned by row
  `MODEL-MM-QWEN4-EXP` under
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978), which is where the
  row's other device-arm debts sit. #2110 closes with this change and so
  cannot carry it.
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
- **W5b-2 (#2123) lands the gated-residual DEVICE ARM, and it lands UNREACHED.**
  `vt::Qwen4ExpGatedResidual` and `vt::Qwen4ExpGatedResidualWriteBack`
  (`include/vt/ops.h`, dispatchers in `src/vt/ops.cpp`, CPU kernels in
  `src/vt/cpu/cpu_qwen4_exp.cpp`) are the hyper-connection stream in `vt::`
  ops, batched over T tokens. Nothing calls them from a production entry point
  at their merge commit: `Qwen4ExpTextModel::Forward` does not exist, and
  `ForwardQwen4ExpForConditionalGeneration` still refuses by name. The wiring is
  owed by **W5b, the forward**, under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031), and the campaign row
  `MODEL-MM-QWEN4-EXP` tracks it under
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978). That is the same
  arrangement W2 (#1987), W3 (#1988) and W4 (#1991) landed under, and for the
  same reason: the only production entry point this architecture has is
  `ModelRegistry::Forward`, which is all-or-nothing, so every slice below the
  whole forward is unreached by construction.

  **Why an op and not a composition**, recorded because the alternative is the
  first thing a reviewer should ask about. Surveyed at `331eda888`: there is no
  ungated per-group RMS norm (`vt::RmsNormGated` has no `group_size`,
  `vt::RmsNormGatedGroup` requires a non-nullable SILU gate), `vt::RmsNorm`
  cannot carry a per-group weight (one `[H]` gamma per row against `hc_norm`'s
  `[hc*H]`), and there is no standalone `silu`, no standalone `sigmoid`, no
  elementwise binary multiply and no axis reduction anywhere in the op set —
  `kSilu`/`kSigmoid`/`kMul` exist only as `FOp` opcodes inside a `constexpr
  FusedRecipe` and are unreachable as free functions. A composition would need
  five new general ops and would still materialise the `[T, hc, H]` broadcast the
  rank-1 write-back exists to avoid. `kDeepseekV4Mhc` is the in-tree precedent.

  **Still owed from this wave, in order:**
  * **The CUDA arm.** Nothing is registered for any device but `kCPU`, so the
    dispatcher refuses by name everywhere else rather than falling back. It was
    not written because it could not be gated: this wave ran on a CPU host with
    no lease, and an ungated kernel is worse than an absent one.
  * **The reduction width that CUDA arm has to choose.** The per-group sum of
    squares accumulates in `double` here, and the W5b-2 table measures a 571x
    separation from a `float` accumulator at group size 2560. A straight
    f32-accumulate block reduction will not meet `kAccumBound`, so the device
    kernel must accumulate wider than f32 or be gated against the oracle
    directly. That is the same decision `## Owed` already records for the W3
    reference, arriving now with a number attached.
  * **One deliberate divergence from upstream, in the bf16 arm.**
    `Qwen4ExpTextRMSNorm.forward` is `self._norm(x.float()) * (1.0 + weight.float())`
    followed by `.type_as(x)`, so on a bf16 stream upstream ROUNDS the normed
    value before the down projection and before the `mixed_input` product. This
    op does not: it widens on load, computes in f32 and rounds once on the store,
    which is this tree's house contract and what `vt::RmsNorm` says of itself in
    the same terms. The f32 arm the goldens are dumped at is unaffected. What
    this costs is that a bf16 parity comparison against a running oracle carries
    a bf16-eps term an f32 one does not, and mutation M13 shows the two are
    distinguishable (reproducing upstream's rounding reds 8 assertions). Owed:
    that term stated in whatever first compares a bf16 arm to the oracle.
  * **A fused single-pass kernel.** The CPU kernel walks the stream four times
    per token (norm, down, up, collapse) plus once per injection row. That is
    correct and slow, and it is deliberate at a wave whose `## Gates` admit no
    speed number; the G4 axis is where it becomes a question.
  * **`hc_norm`'s provenance check.** The op takes vLLM's `1 + w_hf`
    parameterization and never adds 1. `vllm::qwen4_exp::HcNormWeightFromHf` is
    the one home of the fold and a `qwen4exp` GGUF carries it already, so the
    loader owes the cheap check this spec already describes — an unfolded gamma
    is zero-centred and a folded one is centred on 1.0 — at the point where the
    forward binds weights to this op.

- **The epsilon placement was ungated on the DEVICE arm, and W5b-2 closed it
  there and strengthened the host arm in the same flow.** Found while mutating
  the device kernel: moving `+ eps` outside the rsqrt SURVIVED every GOLDEN case
  on both arms, because A, B and C all draw the stream at `hyper_scale = 1.7`,
  where the mean square is O(1) and `eps = 1e-6` moves the answer by about a
  third of `kTol`; `Variant` in `test_qwen4_exp_hc.cpp` has no flag for it
  either. **W3's host suite nonetheless already gated the placement**, through a
  deliberate large-eps probe it shipped for exactly this reason
  (`test_qwen4_exp_hc.cpp:268-276`, whose own comment says a case at an eps large
  enough to separate the two spellings "is the only thing that gates it"): it
  compares `GroupedRmsNorm(..., 4.0f)` against `NormRefD(..., 4.0, Variant{})`,
  and at `big_eps = 4.0` the two spellings are 0.802185 apart against a `kTol` of
  1e-5. MEASURED on a reconstructed pre-repair tree (W3 host suite and goldens at
  `origin/main`, no case D) with the eps mutation applied to
  `qwen4_exp_hc.cpp`: **RED, 2 of 14 cases, 2 assertions** — the `big_eps` probe
  at :276 and golden case B at :343, whose own `eps = 1e-5` puts it marginally
  over tolerance. So the hole was the device arm's alone. Case D closes it there
  and additionally sharpens the host arm from 2 red assertions to the 10 M16
  records at head, which is an enhancement rather than a hole closed. The fourth
  golden case is drawn at `hyper_scale = 0.01`, generated from the same
  lane-pinned oracle source by the same script, and driven by BOTH suites; A, B
  and C regenerate byte-identically. Mutations M7 (device) and M16 (host) are the
  red-after measurement. The defect itself was fixed in the flow that found it,
  which is what AGENTS.md asks for a small and clear fix, so its record is this
  entry and the W5b-2 mutation table. One residual outlived that flow: the
  `#2123` row in `.agents/issue-index.md` still states that the placement
  survived in the W3 host suite, and a row there can never be edited because
  `merge=union` duplicates an edited row instead of merging it.
  [#2141](https://github.com/mudler/vllm.cpp/issues/2141) tracks that residual
  and is the key of the appended row that supersedes the claim; it needed an
  issue number of its own because `check-agent-record.py` refuses a second row
  keyed on `#2123`, reporting it as the duplicate two branches appending the
  same issue would produce.

- **W5c-1, the KV-cache spec, has LANDED**, and this bullet is what it
  replaces rather than a claim it is still owed. It publishes three groups (see
  `### The KV-cache spec is THREE groups and ONE uniform recurrent group`) and
  `MakeQwen4ExpKVCache` returns instead of refusing. The engine blocker this
  bullet named — the runner's `shapes.size() == 2` refusal — was closed by
  `ENG-RECURRENT-MULTISTATE`
  ([#2131](https://github.com/mudler/vllm.cpp/issues/2131), `f7710c1b4`), and
  the SECOND half it named, more than one recurrent group, turned out **not to
  be on this row's path at all**: upstream declares one recurrent shape
  model-wide, so `qwen4_exp` publishes ONE uniform recurrent group and a scalar
  `gdn_group_id_` carries it. The naming correction this bullet recorded is now
  fixed in the source it was about, in flow, as
  [#2198](https://github.com/mudler/vllm.cpp/issues/2198): W4's two comments in
  `src/vllm/model_executor/models/qwen4_exp_qsa.h` cited a `tokens_per_state`
  field with ZERO hits over the pinned vLLM tree and anchored it at an
  unrelated function; they now cite `compress_ratio`
  (`vllm/v1/kv_cache_interface.py:386`, `:393-395`, `:424-435`). The LOCAL
  `QsaSideCacheSpec::tokens_per_state` keeps its name — its arithmetic is right
  and pinned — with a comment saying it has no upstream referent.
- **THE N-GRAM HISTORY IS ZERO-SEEDED, AND 0 IS A VALID TOKEN ID. Nothing in
  this tree can see it.** `CacheBuffer` zero-fills every recurrent state it
  allocates (`src/vllm/v1/worker/gpu/runner.cpp`, both the host and the
  device-`Memset` arm), which is correct for every float state — zero bytes are
  `+0.0f` — and WRONG for the n-gram token history. Upstream's own
  `update_conv_state` pads with 0 too, and the model works around it with an
  explicit EOS left-pad; `PleSequenceState::Reset`
  (`src/vllm/model_executor/models/qwen4_exp_ple.h`) says so in terms: "Pad with
  EOS, never with zero." The forward must therefore EOS-seed that row on the
  same `prefill_has_initial_state == 0` predicate the GDN temporal state already
  uses (`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1513-1514`,
  mirrored at `include/vllm/model_executor/models/qwen3_5.h`). W5c-1 publishes
  the state and CANNOT seed it, because there is no `Qwen4ExpTextModel::Forward`
  to seed it in — that is W5b. **No gate here can catch a zero seed**: token id
  0 hashes to a valid table row, so the model produces fluent wrong text, and
  the only oracle that would catch it is a transformers run this row cannot
  stand up (`gateable = no`). Owned by W5b under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031). Written down rather
  than solved, and it is the single most expensive thing in this section.
- **CLOSED by W5c-2 ([#2249](https://github.com/mudler/vllm.cpp/issues/2249)
  item 3): group 2's block table is gathered. What replaces it is NARROWER and
  it is still `## Owed`: NOTHING READS IT.** The entry this replaces said
  `GPUModelRunner::gather_block_table` is called for `full_attn_group_id_` and
  `gdn_group_id_` and for no other group, so the indexer group's per-request
  block rows never reach a forward. `GPUModelRunner::gather_group_block_tables`
  now gathers EVERY published group's table on the multi-cache path and
  publishes them by GROUP ID on `MultiKvCacheIndex`, mirroring upstream's
  per-group metadata loop (`vllm/v1/worker/gpu_model_runner.py:2551-2567` @ pin
  `5559679229`, `cm.block_table_tensor = _get_block_table(kv_cache_gid)`), so
  the map from a logical position to the physical page now reaches the forward.
  The half that remains is the CONSUMER: `Qwen4ExpQsaPagedCaches::index_key` is
  still a contiguous `[max_kv, indexer_head_dim]` tensor — W5d-3 paged the K/V
  half of the axis and deliberately did not page the side cache — so nothing
  turns that map into a read. Named under all four "Nothing lands dead"
  conditions: what is unreached is the per-group block-table channel's VALUE —
  the runner's gather runs on the production `execute_model` path and the
  refusal in `ModelRegistry::Forward` reads its count, but no forward consumes
  the tables, because `ForwardQwen4ExpForConditionalGeneration` refuses by name;
  the row that owns the wiring is `MODEL-MM-QWEN4-EXP` at **W5** (the layer loop
  itself, which is what the W5a-W5d prerequisite waves feed; W5b-1..6 are all on
  `main`, so naming W5b here would send the reader to finished waves); the
  issues that track it are
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031) and
  [#2249](https://github.com/mudler/vllm.cpp/issues/2249); and it is listed
  here, which is the `## Owed` entry the rule requires. The buffer itself IS
  allocated and gated — `test_runner.cpp`'s
  "a multi-cache topology ALLOCATES its N-state recurrent group" asserts the
  per-group pages — so this is an unread cache and not an unallocated one.
- **EVERY BYTE FIGURE IN THIS ROW IS DERIVED, NOT MEASURED.** On a CPU host
  `kv_cache_backend_resident_` is false
  (`!platforms::GetPlatform(dev.type).is_cpu()`), so the runner takes host
  vectors and nothing on a device has ever held this model's KV. The 3391504 B
  page, the 49.2 MiB uniform slack and the 64 B/token/layer side cache are
  arithmetic over the published shapes, gated as literals, and they are not a
  measurement. Gateable only on `dgx:gpu0`, and `--device cuda` still refuses
  ahead of any tensor I/O for the n-gram expansion
  ([#2083](https://github.com/mudler/vllm.cpp/issues/2083)).
- **`--kv-cache-dtype fp8` now refuses the WHOLE model**, and that is a
  consequence of publishing an MLA group rather than a defect of it.
  `ApplyCacheDType` refuses any `MLAAttentionSpec`
  (`src/vllm/v1/kv_cache_interface.cpp`, `RetypeAttentionSpec`) because upstream
  gives an MLA page its own quantized formula (`fp8_ds_mla`,
  `kv_cache_interface.py:398-410`) and this tree has the formula with no
  fp8_ds_mla store or read. Gated as an executable consequence in
  `test_qwen4_exp_kv_cache.cpp` rather than left to be discovered from a command
  line. The fp8_ds_mla read/write side is NOT this row's; `auto` is unaffected
  and is the production default.
- **The >2048-token QSA gate still has no forward to run.** `## Gates` requires
  a QSA correctness gate past `indexer_budget` tokens of context, because below
  it every candidate block is selected and a pooled-key defect is invisible.
  W5c-1 publishes the cache that gate needs and decodes nothing. Owed by W5b.
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
- **W5b-6 (#2218) RESOLVES THE GAMMA POLARITY, AND IT RESOLVES IT THE OTHER WAY
  ROUND FROM WHAT THAT ISSUE PROPOSED.** #2218 asked the layer loop to fold
  `hc_norm`, `norm_key`, `norm_query` and `norm_conv` through
  `vllm::qwen4_exp::HcNormWeightFromHf` before use. Folding the last three would
  have been the same defect moved one tensor to the left: their consumers
  already add the 1. Measured in this tree rather than argued —
  `RunQwen4ExpQsaBlock` normalizes THREE of its four QSA gammas —
  `idx_q_norm`, `q_norm` and `k_norm` — through `vt::RmsNorm(..., gemma = true)`,
  which is `out * (1 + w)` (`qwen4_exp_qsa_block.cpp:383-384`, `:425-426`,
  `:441-442`, three line pairs for three gammas). **The fourth, `idx_k_norm`,
  never reaches `vt::RmsNorm` at all**: it is handed to `Qwen4ExpQsaIndex`
  (`:401-403`) and consumed inside it by `vt::Qwen4ExpQsaCompress` (`:181`),
  which documents `k_norm_weight` as "the HuggingFace gamma, applied as
  `(1.0 + weight)` ... NOT vLLM's `out * weight`". The polarity is the same
  either way, which is why the conclusion below is unaffected, but the CONSUMER
  is a different op and this entry said `vt::RmsNorm` for all four until the
  W5b-6 review counted the citations against the claim. The PLE host reference
  spells `(1.0 + static_cast<double>(weight[base + i]))` inline at
  `qwen4_exp_ple.cpp:72`. **Three of the four consumers were already on the
  loader's convention and only `vt::Qwen4ExpGatedResidual` was not**, so the op
  moved rather than the loader. The rule is now one line: every gamma in
  `Qwen4ExpWeights` is the RAW HuggingFace parameter and every consumer adds the
  1, `linear_attn.norm.weight` excepted because the converter never folds it and
  `vt::RmsNormGated` wants a plain multiplier. That is also upstream verbatim,
  `Qwen4ExpTextRMSNorm.forward` = `output * (1.0 + self.weight.float())` over a
  zero-initialised parameter (`modeling_qwen4_exp.py:173-178`).
  `HcNormWeightFromHf` survives as the bridge to the W3 HOST reference, whose
  `GroupedRmsNorm` keeps vLLM's `out * w` form, and it is now called from the
  two suites that drive that reference and from no production path.
- **W5b-6 (#2218) LANDS UNREACHED, by AGENTS.md "Nothing lands dead".**
  `vt::Qwen4ExpGatedResidual` and `vt::Qwen4ExpGatedResidualWriteBack`
  (`include/vt/ops.h`, dispatchers `src/vt/ops.cpp`, CPU kernels
  `src/vt/cpu/cpu_qwen4_exp.cpp`) are the ops whose gamma contract this wave
  changed, and at its merge commit nothing calls either from a production entry
  point. Their only call sites are `tests/vllm/models/test_qwen4_exp_hc_device.cpp`
  and the new `tests/vllm/models/test_qwen4_exp_forward.cpp`. That second suite
  reaches the PRODUCTION LOADER — `ModelRegistry::Load` over a `qwen4exp` file —
  and it is what makes the fix gateable at all, but a test driving a production
  loader is still a test: it is not a production entry point, and reaching the
  loader does not reach the op. `Qwen4ExpTextModel::Forward` does not exist and
  `ForwardQwen4ExpForConditionalGeneration`
  (`src/vllm/model_executor/models/qwen4_exp_registry.cpp`) still refuses by name
  before any downcast, so the op stays unreached for exactly the reason W5b-2
  (#2123) recorded when it landed the op in the first place. The wiring is owed by
  **W5b, the layer loop**, under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031), owned by row
  `MODEL-MM-QWEN4-EXP` and tracked by campaign
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978); the five measured
  prerequisites that wave must clear first are the entry below this one.
- **THE LAYER LOOP'S PREMISE — "every component it composes is already on
  `main`" — IS FALSE, AND HERE ARE THE FIVE THINGS THAT ARE NOT.** Surveyed
  against this tree while W5b-6 was in flight, each independently sufficient to
  stop a token, and each now named in the `ForwardQwen4ExpForConditionalGeneration`
  refusal so the next reader does not re-derive it:
    1. **CLOSED by W5d-1 (#2249 item 1): there is no standalone grouped RMS
       norm — there is now.** The refusal string that still said otherwise is
       [#2288](https://github.com/mudler/vllm.cpp/issues/2288), fixed by
       [#2265](https://github.com/mudler/vllm.cpp/pull/2265). `vt::RmsNormGroup` / `OpId::kRmsNormGroup` is that
       op, registered on `kCPU`, gated by `tests/vt/test_ops_rms_norm_group.cpp`
       at 7 cases / 69 assertions, and its own mutation record is the
       `## Mutation record — W5d-1` section above. The survey text is kept below
       because it is the argument that produced the op and the layer loop still
       has to CALL it, which nothing does yet. `Qwen4ExpTextPLELayer` holds
       three `Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=hidden_size)` —
       `norm_key`, `norm_query`, `norm_conv` — reducing over `hc` independent
       `hidden_size` slices of the 10240-wide stream. `include/vt/ops.h:556`
       states the gap in its own words: "`kRmsNormGated` has no group_size;
       `kRmsNormGatedGroup` requires a SILU gate". `vt::RmsNorm` reduces the
       whole last dim and takes a `[D]` gamma, and the PLE gamma is `[hc*H]`, so
       it cannot express this even per-branch. The only grouped reduction in the
       tree is FUSED inside `vt::Qwen4ExpGatedResidual` and is not exposed.
       **This is new op work, and it is the same "why a fused family op" argument
       W5b-2 made, arriving at the opposite answer because PLE needs the norm
       without the mix.**
    2. **CLOSED by W5d-3 (#2249 item 2), for the K/V half only: the QSA
       consumer is CONTIGUOUS and the published cache is PAGED — it now reads
       the paged one.** `Qwen4ExpQsaPagedCaches` and `RunQwen4ExpQsaBlockPaged`
       bridge KV group 0 (the `FullAttentionSpec`) through a
       `kv_block_table`/`kv_block_size` address mode inside the same
       `vt::Qwen4ExpQsaGatherAttention`, rather than a second op. The INDEXER
       side cache is untouched and still contiguous, which is item 3 below and
       is owed as W5c-2 — so this item is closed and item 3 is not, and the two
       are the SAME axis split in half. The survey text follows, because it is
       the argument that produced the wave and the layer loop still has to CALL
       the paged arm, which nothing does.
       `Qwen4ExpQsaCaches` is `key`/`value` `[max_kv, num_kv_heads, head_dim]`
       and `index_key` `[max_kv, indexer_head_dim]`
       (`qwen4_exp_qsa_block.h`), while `MakeQwen4ExpKVCache` publishes a
       `FullAttentionSpec` and an `MLAAttentionSpec` the runner allocates as
       paged `CacheBuffer`s. Bridging them is a paged arm of
       `RunQwen4ExpQsaBlock`, not a cast.
    3. **CLOSED by W5c-2 (#2249 item 3): group 2 was allocated and unread —
       its block table is now gathered.** This item said `gather_block_table`
       has three call sites and reaches exactly `full_attn_group_id_` and
       `gdn_group_id_`. It has a FOURTH now,
       `GPUModelRunner::gather_group_block_tables`, which runs over every
       published group on the multi-cache path and publishes the tables by
       group id on `MultiKvCacheIndex`. What the loop still needs from this
       axis is the CONSUMER: item 2's paged arm reads the K/V through a block
       table and the INDEXER side cache off a contiguous `[max_kv, D]` array,
       so the map this item delivers has no reader. That stays carried under
       `## Owed` above.
    4. **The MoE weights need an adapter.** `Qwen4ExpMoeWeights` holds stacked
       `gate_exps`/`up_exps` `[E, moe_I, H]` and `down_exps` `[E, H, moe_I]`;
       `RunMoeBlock` reads `MoeBlockWeights`, whose arms are per-expert
       `[H, I]` vectors, an `Nvfp4Weight` set, or the stacked keep-quant
       `expert_gate_kq [E*I, H]` / `expert_down_kq [E*H, I]`. **CLOSED by W5d-4
       ([#2249](https://github.com/mudler/vllm.cpp/issues/2249) item 4),
       `src/vllm/model_executor/models/qwen4_exp_moe.{h,cpp}`. The sentence that
       used to stand here — "the third arm's shapes are exactly the qwen4_exp
       ones", so the adapter is "a reinterpretation … rather than a copy" — is
       measured FALSE, and it is false on the arm every shipped checkpoint
       takes.** `LoadStackedExperts` records the tower as RANK 3 `[E, N, K]`
       (`qwen4_exp_weights.cpp:160-164`) and `MoeBlockWeights::expert_*_kq` is
       RANK 2 `[E*N, K]`; the default keep-quant route
       (`Qwen35GroupedMoeEnabled`, ON) hands that tensor to
       `vt::MatmulBTQuantGrouped`, whose first check is
       "matmul_bt_quant_grouped: rank-2 out/act/weight required"
       (`src/vt/ops.cpp:223`). **That refusal is ROUTE-CONDITIONAL and the
       sentence above is scoped to the default route on purpose.** With
       `VT_QWEN35_GROUPED_MOE=0` the seam takes the per-expert `ExpertMlpKq`
       path, which reaches `KqResidentSlice` (`qwen3_5.cpp:5665-5678`); that
       helper rebuilds a rank-2 view from its `N`/`K` ARGUMENTS by pointer
       arithmetic, sets `wt.rank = 2` itself and never reads the tower's
       declared rank, so a rank-3 tower does not throw there — and, the tower
       being contiguous `[E, N, K]`, `row_off = e * N` lands on exactly the
       right slice, so it answers correctly. #2249 item 4's original sentence is
       therefore literally true of the NON-default route. It is false of the one
       every shipped checkpoint takes, which is why this wave was the size it
       was. Three more differences a shape comparison cannot
       see: the router and shared gate are f32 by `LoadMoe`'s deliberate choice
       and the CUDA GEMM refuses a (bf16, f32) pair by name
       (`cuda_matmul.cu:397-403`), so passing them through runs on CPU and dies
       on every GPU; `MoeBlock` selects the whole expert path from
       `expert_gate_kq` ALONE, so a per-tensor residency split reads as
       keep-quant and dereferences an empty tower; and a **bf16** tower cannot
       use the stacked fields at all (`ops.cpp:231` refuses a non-block dtype),
       so the bf16 arm fills the PER-EXPERT vectors — with zero-copy borrowed
       views, because three copies per layer at the released geometry is 240 GB
       across the stack. Both arms are now gated against a from-scratch
       reimplementation of the lane-pinned oracle in
       `tests/vllm/models/test_qwen4_exp_moe.cpp`. **And the alternate route is a
       measured RESULT rather than an admitted limit.**
       `Qwen35GroupedMoeEnabled()` caches in a function-local `static const`
       (`qwen3_5.cpp:6299-6302`), which prevents flipping it MID-PROCESS, not
       before launch — so `VT_QWEN35_GROUPED_MOE=0` does exercise `ExpertMlpKq`.
       Run that way, both value cases pass at BIT-IDENTICAL `max|diff|` to the
       default route (bf16 `0.00218359`, keep-quant `0.00865547`), which is the
       seam's own byte-identity claim at `qwen3_5.cpp:7261` measured rather than
       inherited. The suite runs on both routes and says which behaviour it is
       asserting on each.
    5. **CLOSED by W5d-2 (#2249 item 5, `3ed2378a3`): the mRoPE table builder
       has internal linkage — it no longer does.** This half of
       [#2288](https://github.com/mudler/vllm.cpp/issues/2288) is pre-existing
       debt from [#2264](https://github.com/mudler/vllm.cpp/pull/2264), which
       corrected this prose and left the refusal string; both are fixed by
       [#2265](https://github.com/mudler/vllm.cpp/pull/2265). `BuildMropeCosSinHost` is
       declared at `include/vllm/model_executor/models/qwen3_5_mrope.h:55` and
       defined without `static` at `qwen3_5.cpp:9475` on this merged head, so
       the QSA half can build the SAME tables the Qwen3.5/3.6 VL drivers build.
       The call from the loop is still owed, and it is W5b's. The survey text
       follows.
       `BuildMropeCosSinHost` WAS `static` at `qwen3_5.cpp:9472`, and
       `RunQwen4ExpQsaBlock` needs BOTH layouts derived from it: the packed
       bf16 `[P, rot]` `cos|sin` cache `vt::RopeFromCache` reads and the two
       separate f32 `[P, rot]` tables `vt::Qwen4ExpQsaCompress` reads, with
       `CheckRopeLayoutsAgree` verifying they describe the same angles.
  **And one more that is not this row's:** `ModelRegistry::Forward` refuses ANY
  non-null `multi_kv` by name (`model_registry.cpp:462-478` — `:428-440` on
  W5c-2's parent; W5c-2's own `BlockTableForGroup` sits above it and moved it,
  and the refusal now also reports how many groups have a gathered table), and
  this model's three published groups make the runner set it
  (`runner.cpp:787-804`, `:2325`). A forward reached through
  `ModelRegistry::Forward` with a hand-built positional cache set is gateable
  today; a forward reached through `GPUModelRunner` is not, and lifting that
  refusal is an engine seam change DeepSeek-V4 waits on too.

### The llama.cpp G4 denominator, owed after the arm landed

The `llama-cpp-qwen4exp` oracle is `gateable = yes` and its harness is committed,
and three things are still owed before a single cell of that table exists.

- **[#2261](https://github.com/mudler/vllm.cpp/issues/2261): a
  `KV_BYTES_PER_TOKEN` measured on a leased load, without which the ladder
  refuses.** `llama-server` at the pin reports no KV size — measured on the row's
  own production capture, which is the complete unfiltered server output — so a
  guard that defaulted the term to zero was weightless on the only server this
  harness will face. It now refuses (`E_KV_UNREPORTED`, 21) rather than sizing a
  49,152-token context over 32 slots against a 67.5 GiB model with nothing
  bounding the cache. A refusal is the correct output until the number exists.
- **[#2262](https://github.com/mudler/vllm.cpp/issues/2262): the mutation sweep
  is not re-executable and the CUDA toolchain is asserted rather than pinned.**
  Nine of eleven recorded mutations have no committed driver, and
  `cuda-toolkit-13-0` names a channel rather than a version; `EXPECT_NVCC` makes
  the drift refuse but does not make apt serve one build.
- **The ladder run itself, and a vllm.cpp arm to put beside it.** There is
  nothing to compare until `ModelRegistry::Forward` stops refusing `qwen4_exp` by
  name, and the vllm.cpp cell must be a TEXT-ONLY configuration on the same
  UD-IQ1_S artifact, because the denominator reports every modality false.


## Now

`ACTIVE`. Nine reviewed waves have landed. Eight of them are unreached by design
and the ninth, W5a, is the only one with a production call site:

| Wave | Lands | Issue |
|---|---|---|
| W1 | the config layer: `qwen4_exp` resolves, parses and VALIDATES | [#1981](https://github.com/mudler/vllm.cpp/issues/1981) |
| W2 | the hashed n-gram index and the PLE dilated depthwise conv | [#1987](https://github.com/mudler/vllm.cpp/issues/1987) |
| W3 | the 4-branch gated-residual hyper-connection stream | [#1988](https://github.com/mudler/vllm.cpp/issues/1988) |
| W4 | Qwen Sparse Attention with a GATHER consumer | [#1991](https://github.com/mudler/vllm.cpp/issues/1991) |
| W6a | IQ4_NL, Q5_0 and a dequantizing gather, so the artifact OPENS | [#1989](https://github.com/mudler/vllm.cpp/issues/1989) |
| W5a | the GGUF weight loader, REACHED through the `load_weights` hook | [#2031](https://github.com/mudler/vllm.cpp/issues/2031) |
| W5b-1 | `RunGdnBlockPaged`, the GDN block seam the forward needs cross-TU | [#2110](https://github.com/mudler/vllm.cpp/issues/2110) |
| W5b-2 | the gated-residual hyper-connection stream as two `vt::` ops | [#2123](https://github.com/mudler/vllm.cpp/issues/2123) |
| W5b-3 | the PLE dilated depthwise causal conv as `vt::Qwen4ExpPleConv` | [#2156](https://github.com/mudler/vllm.cpp/issues/2156) |
| W5b-4 | Qwen Sparse Attention as two `vt::` ops, plus the unmapped-tail probe | [#2167](https://github.com/mudler/vllm.cpp/issues/2167) |
| W5b-5 | `Qwen4ExpTextAttention` as ONE block, and the indexer composition in `src/` | [#2211](https://github.com/mudler/vllm.cpp/issues/2211) |
| W5c-1 | the KV-cache spec: THREE groups, REACHED through `make_kv_cache` | [#2031](https://github.com/mudler/vllm.cpp/issues/2031) |
| W5d-2 | `BuildMropeCosSinHost` loses `static`: ONE mRoPE table builder, cross-TU | [#2249](https://github.com/mudler/vllm.cpp/issues/2249) |

**Reached, and LOADING — on a CPU device:** a `qwen4exp` file lands on
`Qwen4ExpHfConfigFromGguf` through the `kGgufArchArms` dispatch row, the registry
resolves `Qwen4ExpForConditionalGeneration`, and W5a's `load_weights` hook now
materializes the whole text tower instead of refusing. That is the first
production call site this row has had. On any device that cannot gather from a
block table the load REFUSES BY NAME ahead of any tensor I/O, because the
n-gram table would otherwise expand from 26.822 GiB to 95.368 GiB of host
memory ([#2083](https://github.com/mudler/vllm.cpp/issues/2083)); the CUDA
gather arm is owed.

**Reached, and no longer refusing: the KV-cache spec.** W5c-1
([#2031](https://github.com/mudler/vllm.cpp/issues/2031)) makes
`make_kv_cache` return three groups — the QSA layers' paged K+V, ONE uniform
recurrent group carrying `[gdn_conv, temporal, ple_conv, ngram]` on every
linear layer, and the QSA indexer side cache as an `MLAAttentionSpec` at
`compress_ratio` 4 — over real per-layer names, so the runner takes its
multi-cache path and allocates every published cache. The engine half was
`ENG-RECURRENT-MULTISTATE` (#2131); the second half that row expected to be
needed, more than one recurrent group, is NOT on this path, because upstream
declares one recurrent shape model-wide. Three things it does not do, each under
`## Owed`: the n-gram history is ZERO-SEEDED where it needs EOS and no gate here
can see that, nothing READS the side cache's block table yet (W5c-2 made the
runner gather it and hand it to the forward; no consumer takes it), and every
byte figure is derived on a CPU host rather than measured on a device.

**Reached, and still refusing:** the forward. Nothing decodes a token, so there
is still no token number, no speed number, no `examples/server` e2e and no
`docs/USAGE.md` weights row — that row is owed in the same change that makes an
arm SERVE, which is W5b, not W5a. W2, W3 and W4
remain host reference math with no production call site.

**W5b-6 ([#2218](https://github.com/mudler/vllm.cpp/issues/2218)) closes the
gamma polarity and it does NOT decode.** `vt::Qwen4ExpGatedResidual` now takes
the RAW HuggingFace gamma and adds the 1 itself, which is the convention the
other three consumers of this architecture's gammas already had, so the layer
loop can hand it `Qwen4ExpWeights` directly instead of scaling the
hyper-connection stream by ~0. The gate is
`tests/vllm/models/test_qwen4_exp_forward.cpp`, the first **`qwen4_exp`** suite
that LOADS a gamma through `ModelRegistry::Load` and runs it through a device op
in one case — which is why eleven single-sided waves of THIS row could not see
it. **THE UNSCOPED FORM OF THAT SENTENCE WAS FALSE AND IS CORRECTED HERE.** It
claimed the first such suite in the tree; it is not.
`tests/vllm/models/test_nemotron_h_paged_forward.cpp` and
`tests/vllm/models/test_kimi_linear_paged.cpp` each call `ModelRegistry::Load`
inside a `TEST_CASE` and drive the loaded weights, gammas included, through the
device ops of a forward. The claim that survives is the narrow one, and it is
the one the argument needed: no `qwen4_exp` suite had ever composed the loader
with an op, so the contradiction between them was unreachable here. The synthetic
`qwen4exp` file moved to `tests/support/qwen4_exp_gguf_fixture.h` so the loader
suite and the forward suite share ONE builder.

**AND IT CORRECTS THIS SECTION'S OWN CLAIM.** The paragraph below used to say
"THE OP AND SEAM WORK IS FINISHED; WHAT IS LEFT IS THE LAYER LOOP." That is not
true. Five things the loop composes were absent from `main` when W5b-6 surveyed
it — a standalone grouped RMS norm for PLE's three norms, a PAGED QSA consumer,
the group-2 block table (W5c-2, landed: the runner gathers every published
group's table), a MoE weight adapter, and an externally linked mRoPE builder — and `ModelRegistry::Forward` additionally refuses every
multi-cache topology by name, which is what this model publishes. Each is
measured and cited under `## Owed`, and the production refusal in
`qwen4_exp_registry.cpp` now names them instead of naming W2, W3 and W4, which
landed. **A wave dispatched to "write the layer loop" will not decode a token;
it has these prerequisites, at least two of which (the grouped norm, the paged
QSA arm) are op-sized waves of their own.**

**ALL FIVE ARE NOW CLOSED, AND THE COUNT IS STATED HERE RATHER THAN LEFT TO A
READER TO RECOUNT.** The stale enumeration is
[#2288](https://github.com/mudler/vllm.cpp/issues/2288), filed for traceability
and FIXED IN THE SAME FLOW by
[#2265](https://github.com/mudler/vllm.cpp/pull/2265), the wave this correction
first rode with.

- **Item 1, the grouped RMS norm**, is `vt::RmsNormGroup`, landed by W5d-1
  ([#2249](https://github.com/mudler/vllm.cpp/issues/2249) item 1) as
  `25ee19464`.
- **Item 5, the externally linked mRoPE builder**, is `BuildMropeCosSinHost`,
  landed by W5d-2 (#2249 item 5) as `3ed2378a3`; that wave corrected the
  paragraph above and did NOT correct this list or the production refusal
  string, so both had been naming a finished seam since it merged.
- **Item 4, the MoE weight adapter**, is
  `src/vllm/model_executor/models/qwen4_exp_moe.{h,cpp}`, landed by W5d-4 (#2249
  item 4) as `3f9177f7f`. Closed as a SEAM, not as a call: W5d-4 landed
  unreached and says so under `## Owed`, exactly as W5d-1 does.
- **Item 2, the PAGED QSA consumer**, is `Qwen4ExpQsaPagedCaches` +
  `RunQwen4ExpQsaBlockPaged` over a `kv_block_table`/`kv_block_size` ADDRESS
  MODE inside `vt::Qwen4ExpQsaGatherAttention`, landed by W5d-3 (#2249 item 2) —
  the wave this section is being merged with, which is why this recount rides
  here. Closed as a SEAM as well, and only for the K/V half: the INDEXER side
  cache is still contiguous, and its paged STORE outlives the survey — see
  below, where it is named as a gap that was never one of the five.
- **Item 3, the group-2 block table**, is
  `GPUModelRunner::gather_group_block_tables`, landed by W5c-2 (#2249 item 3) —
  the wave THIS section is being merged with, which is why this recount rides
  here a second time. Every published group's table is gathered on the
  multi-cache path and published by group id on `MultiKvCacheIndex`. Closed as
  a SEAM, like the other four: the MAP reaches the forward and no consumer
  reads it, which is the narrower entry `## Owed` now carries in its place.

**NONE remain. The count is ZERO.** What refuses is no longer a prerequisite:
it is the LAYER LOOP itself, `Qwen4ExpTextModel::Forward`, owned by
[#2031](https://github.com/mudler/vllm.cpp/issues/2031), which nothing above
substitutes for. Two things sit beside it and neither is one of the five. The
QSA indexer side cache's PAGED STORE is still owed — the map arrives with W5c-2
and `Qwen4ExpQsaPagedCaches::index_key` is still contiguous, so the loop wave
inherits that store rather than finding it done. And the `multi_kv` refusal is
not this row's: `ModelRegistry::Forward` refuses every multi-cache topology by
name and this model publishes one, which #2249 records as belonging to an
engine row. The refusal in `qwen4_exp_registry.cpp` says exactly this at this
merge commit, and the emitted bytes were read back out of the running hook to
prove it.

**AND THE COUNT IS ZERO BY SET DIFFERENCE, WHICH IS NOT WHAT EITHER SIDE OF
THIS MERGE SAID ON ITS OWN.** `main` carries W5d-3, whose text removed item 2
and still listed the group-2 block table that W5c-2 closes; this branch's text
removed item 3 and still listed the paged QSA consumer that W5d-3 closed. Both
sides therefore said ONE, both were exactly one item too long, and taking either
side whole would have landed a survey naming finished work — #2288 again, in its
sixth turn on this row. Deleting the enumeration instead would have been the
opposite error, because a survey that stops counting reads as an unfinished one.
The same trap has now caught this branch three times, against three baselines:
against #2265, whose edit renumbered five to THREE while still listing the paged
consumer; against W5d-4, which renumbered to FOUR from the other side; and here,
against W5d-3. The remaining set is the survey minus EVERY landed wave, never
the shorter of two lists — and when that difference is empty, the survey says so
in words rather than by falling silent.

ONE SHAPE FROM W5d-3'S OWN EDIT SURVIVES THE RECOUNT AND IS WORTH KEEPING.
Its five-to-four step MERGED two items rather than dropping one — a statement
about that edit's arithmetic, not about the count here, which is zero. What
W5d-3 discharges is the K/V half of the paged axis; what survived of it is the
indexer side cache, which was already its own item and whose block table W5c-2
now gathers. A reader who counts items without reading them will conclude a
prerequisite vanished when it was only folded into the neighbour it shares an
axis with — and will now conclude the port is done when what closed is its
prerequisites and not its loop.

**What is owed, in order. THIS PARAGRAPH'S OPENING CLAIM WAS WRONG AND IS
CORRECTED ABOVE: the op and seam work is NOT finished.** What follows is still
the right list of what W5b-1..5 landed; what it got wrong is the inference that
nothing else was missing. W5b needed five slices and four of them are ops or
seams:
`RunGdnBlockPaged` for the 36 linear layers (W5b-1), the two gated-residual ops
for the 10240-wide stream (W5b-2), `vt::Qwen4ExpPleConv` (W5b-3) and the two QSA
ops (W5b-4). W5b-5 turned the last of those into a decoder-layer BLOCK —
`RunQwen4ExpQsaBlock`, the first production composition of the QSA indexer — so
nothing the QSA indexer needs is missing from the `vt::` surface any more —
though the PLE block's grouped RMS norm still is, which the sentence this
replaces overstated into a claim about the whole architecture.
What has no production shape yet is the PLE block, the GDN weight adapter onto
`GdnLayerWeights`, the hyper-connection stream through the per-layer loop, and
the loop itself. TWO entries have left that list, and for the same reason in
both cases: the seam is in `src/` and only the CALL is owed. The MoE weight
adapter onto `MoeBlockWeights` is no longer on it — W5d-4
([#2249](https://github.com/mudler/vllm.cpp/issues/2249) item 4) is
`src/vllm/model_executor/models/qwen4_exp_moe.{h,cpp}`, which composes
`Qwen4ExpTextSparseMoeBlock` through the shared sparse-MoE seam rather than a
second MoE path, and the layer loop still has to CALL it. The mRoPE cos/sin
table build is not on it either — W5d-2
([#2249](https://github.com/mudler/vllm.cpp/issues/2249) item 5) gave
`BuildMropeCosSinHost` external linkage behind
`include/vllm/model_executor/models/qwen3_5_mrope.h`, so the QSA half builds the
SAME tables the Qwen3.5/3.6 VL drivers build rather than a second copy — but the
loop still has to CALL it, and that call is W5b's.
The trap this paragraph used to warn about is FIXED, not pending: the loader
stores every gamma in the RAW HuggingFace parameterization and
`vt::Qwen4ExpGatedResidual` used to want the opposite, so a layer loop handing it
the loaded tensor applied a near-zero scale that reads as a checkpoint bug. W5b-6
([#2218](https://github.com/mudler/vllm.cpp/issues/2218)) moved the op onto the
loader's convention and gated the composition; a layer loop may now hand it
`Qwen4ExpWeights` directly. The
mixer/lm_head tail is not owed: the terminal
`use_combine=false` mixer IS `vt::Qwen4ExpGatedResidual` with a null
`block_inject`, gated as its own case in `test_qwen4_exp_hc_device.cpp`, and
`Qwen4ExpTextModel` has no final RMSNorm after it (`## Owed`), so the tail is
that op plus a `kMatmulBT` lm_head. W5c, the KV-cache spec with three conv states
and the QSA side cache — and note that this tree's runner accepts exactly ONE
`MambaSpec` group whose `shapes` must be exactly two, conv then temporal
(`src/vllm/v1/worker/gpu/runner.cpp`), so the third conv stream a PLE layer needs
is a runner change and not only a registry one
([#2131](https://github.com/mudler/vllm.cpp/issues/2131)); then the first served
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
