# SPEC-DFLASH2 — DFlash2: the grouped dynamic convolution and the candidate selector

**Row:** `SPEC-DFLASH2` (engine-matrix, speculative decoding).
**Issue:** [#1314](https://github.com/mudler/vllm.cpp/issues/1314).
**Predecessor:** [dflash-spec-decode.md](dflash-spec-decode.md) (`SPEC-DFLASH`, DONE).
**Follow-on issue:** [#1628](https://github.com/mudler/vllm.cpp/issues/1628) — the DFlash2 candidate
selector could not consume a QUANTIZED target `lm_head`, which blocked the only speculative arm
`r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` has. `## Risks/decisions` D14 records the decision, and
`## Owed` O28/O29/O30/O31 record what it leaves owed.
**In-flow fix:** [#1575](https://github.com/mudler/vllm.cpp/issues/1575) — W5's `tests/vllm/models/test_qwen3_dflash2_gguf.cpp` called `::getpid()` directly instead of the
portable `tests/support/process_id.h` seam, which held `build-newest-gcc` red on `main` from
`5702d8f83`. Fixed in flow; not owed.
**Kind:** structured spec. No production code lands with this spec; the pull
request shape for this row is SEPARATE spec and implementation pull requests,
recorded at row claim on 2026-08-19.

**DFlash2 is a second architecture, not a change to DFlash.** Upstream leaves the
DFlash draft running untouched and adds `DFlash2DraftModel` beside it. This port
mirrors that polarity: no shipped DFlash behaviour moves, except the one config
rule `## Risks/decisions` D4 records, which upstream changes in the same commit
and which our tree resolves the old way.

**This row is a correctness row first.** DFlash2's claim is acceptance, and
acceptance is invisible to a token gate: a drafter that proposes worse tokens
still emits the target's tokens, because the verify is lossless. Nothing in
`## Gates` accepts a speed number before the acceptance gate reads.

## Scope

In scope: the `DFlash2DraftModel` architecture end to end on the bf16 arm and
the GGUF drafter arm — routing from the draft's own `config.json`, the grouped
dynamic depthwise convolution inside each draft block, the candidate selector
and its device path walk, the vocabulary top-k the selector consumes, and the
`is_causal` resolution the published checkpoint depends on. The draft must be
reachable from `--speculative-config` through the loader and
`ModelRegistry::Forward`, not merely constructible.

BOTH published DFlash2 checkpoints are in scope, because both resolve to the SAME
class: upstream registers one architecture, `DFlash2DraftModel -> qwen3_dflash2`,
and `z-lab/Qwen3.8-27B-DFlash2` and `z-lab/Muse-Glimmer-30B-DFlash2` both declare
`model_type` `qwen3`. What the second one adds is not a class but VALUES:
`block_size` 16 against the first's 8, and two of the three output scalars set
rather than defaulted ([#1327](https://github.com/mudler/vllm.cpp/issues/1327)).
An earlier revision of this section excluded "a second DFlash2 target family",
which was true about the CLASS and false about the work.

Out of scope: any change to the DFlash draft's own behaviour beyond the shared
`is_causal` rule; the DSpark lane; and any throughput claim, which `## Gates`
defers with its reason.

## Upstream chain

**MERGED UPSTREAM on 2026-08-21 at 05:27:22Z**, at head
`3406ec1dae9916f920b90f0dbf90dcf54923d042`, merge commit
`b389ac29465b33f9e9c534df221ea3c129e9793f`.
[vllm-project/vllm#52816](https://github.com/vllm-project/vllm/pull/52816) was
opened 2026-08-18 on base `9842d701`. This row's gates were read against head
`66e5414c6d75a8529473d977f7458c140bbab8a0`.

**THIS SECTION SAID "Not merged upstream" AFTER THE MERGE HAD ALREADY HAPPENED,
and so did four other places in this file, the `#1538` index row, and TWO LIVE
SURFACES OUTSIDE THIS FILE.** The merge landed at `05:27:22Z`; W6's work commit
`bb416e0ae` was authored at `06:13:50Z`, 46 minutes later. Every open-PR
statement W6 wrote was therefore already false when it was written, and W6 did
not re-read the forge before committing. Corrected on 2026-08-21 by the W6
repair wave, which read `gh api repos/vllm-project/vllm/pulls/52816` itself
rather than inheriting the claim. The reconciliation the merge makes due is
[#1561](https://github.com/mudler/vllm.cpp/issues/1561); `## Owed` O21 carries
it.

**THIS IS THE COMPLETE INVENTORY over `src/`, `include/`, `tests/`, `docs/` and
`.agents/`, and the first repair wave's was short.** That wave corrected the five
spec statements, SUPERSEDED the index row by appending a new one, and stopped
there, which left the claim standing in the two places a USER can reach — in
shipped product output and in a public header — for a second review to find. The
five directories are the sweep's scope and are named because the sweep is not the
whole tree; the one occurrence outside them is recorded under the table:

| surface | what it said | state |
|---|---|---|
| 5 statements in this file, at `bb416e0ae` lines 46 (`## Upstream chain`), 332 and 383 (`## Gates`), 464 (`## Work breakdown`) and 1593 (`## Owed` O21) | "Not merged upstream" / "#52816 was still open" / "is still OPEN" | corrected by the first repair wave (`e8cb8d4a3`) |
| the `#1538` row of `.agents/issue-index.md` | "while it is still open" | **SUPERSEDED, not corrected**, by the first repair wave (`e8cb8d4a3`) — see below |
| `src/vllm/entrypoints/model_loader.cpp` — the DFlash2 startup notice, printed on **every** draft load, safetensors and GGUF alike | "which is OPEN upstream at head `66e5414c…`" | corrected by the SECOND repair wave |
| `include/vllm/config/speculative.h` — the `IsDflash2Draft` BEYOND-PIN comment | "is OPEN at head `19c93519…`", the first of three heads, so doubly stale | corrected by the SECOND repair wave |

**SUPERSEDED IS THE WORD, AND "CORRECTED" WOULD BE FALSE.** An earlier revision
of this table said the `#1538` row was corrected by `e8cb8d4a3`, and it was not:
`git show e8cb8d4a3 -- .agents/issue-index.md` is **3 insertions, 0 deletions**,
appending the #1561, #1562 and #1564 rows. The `#1538` row is byte-identical to
its `bb416e0ae` text — sha256 `c48fdd02…` on both — and it still reads "while it
is still open" in the tree today. That is the rule working rather than the rule
being broken: `.agents/issue-index.md` is append-only and a landed row is NEVER
edited, so the only available repair is the one that was made, appending #1561
to say what changed. A reader who wants the state of vllm#52816 reads #1561 and
this table, not the #1538 row.

The `#1561` row in turn states the inventory as "five statements in the spec plus
the `#1538` index row", which is the short one, and it is likewise not edited.
**This table is the authority and neither index row is**; #1561's own owed work
is moving the gate head, which is a separate thing from the inventory of what
said OPEN.

**ONE OCCURRENCE LIVES OUTSIDE THE FIVE SWEPT DIRECTORIES, and it is history
rather than a live claim.** `scripts/check-agent-record.py:637-638` reads
"BEYOND-PIN on vLLM PR 52816 (OPEN at head `19c93519`, base `9842d701`)", which
wraps across the two lines and is not readable at `:637` alone. It is the sole
`52816` in `scripts/`, and it sits inside the dated append-only ratchet
chronology that `b953bfe82` wrote on 2026-08-19 as the `165 since 2026-08-19`
entry: that entry records what was true when the ratchet mark moved, which is a
statement about 2026-08-19 and not about today. It is left as written for the
same reason the `#1538` index row is, and it is recorded here so that a fourth
sweep does not read it as a fifth stale surface. What would change it is the
ratchet arithmetic being wrong, which it is not.

**And the class is now gated rather than re-audited.** The startup-notice case in
`tests/vllm/v1/spec_decode/test_dflash2_runner_reach.cpp` matched only the
substring `"52816"`, so it could not see the word `OPEN` at all and stayed green
across both false versions of the notice. It now asserts the merged wording and
the merged head are PRESENT and both spellings of the open claim are ABSENT, so
a third recurrence in that surface reds on every box with no checkpoint and no
GPU.

**A SECOND gate on the same notice was NOT strengthened, and that is recorded
rather than left to be assumed.**
`tests/vllm/entrypoints/test_dflash2_draft_routing.cpp:257` also asserts on the
startup notice and also matches only the substring `"52816"`, so it is still
blind to the word `OPEN`. It is not a finding: one strengthened gate catches the
CLASS, because a false notice reds `test_dflash2_runner_reach` on every box with
no checkpoint and no GPU, and hardening the second would add a second copy of one
guarantee rather than a second guarantee. The third fresh review classified it
that way. It is written here so that a later reader does not assume both were
hardened.

**THE HEAD MOVED under this row, and W3 is the wave that reconciles it**
([#1404](https://github.com/mudler/vllm.cpp/issues/1404)). W1 and W2 were written
against `19c9351904df4c63042671bc67a866ca48dc7d6f`; every anchor W3 cites was
re-read at the new head, and the anchors W1 and W2 recorded are annotated where
they move rather than silently rewritten. Diffing the two heads changes five
files. **The move is small in the file W3 ports and LARGE in the file W4 ports,
and an earlier revision of this section
said "for the two this row ports it is exactly two things, both infrastructure
rather than math" — true of the first file and FALSE of the second.** Measured
with `git diff --no-index --numstat` over the two raw blobs fetched from
`raw.githubusercontent.com` at each head on 2026-08-20:

| File | Delta between the heads | Whose wave |
|---|---|---|
| `vllm/model_executor/models/qwen3_dflash2.py` | +24 / −11 | W3 |
| `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` | +37 / −34 | W4 |
| `vllm/v1/worker/gpu/spec_decode/speculator.py` (the BASE class) | +24 / −6 | W4 |

In `qwen3_dflash2.py` it IS exactly two things, both infrastructure rather than
math:

- `set_model_tag("dflash2_candidate_selector")` around the selector's
  construction. A DELIBERATE NON-PORT — `## Risks/decisions` D11.
- the LM-head guard widened to accept `UnquantizedLinearMethod` beside
  `UnquantizedEmbeddingMethod`, which is the FOLDED-IN
  [#52883](https://github.com/vllm-project/vllm/pull/52883) (it no longer sits
  stacked; the new head carries it). PORTED — `## Risks/decisions` D12. The head
  also lengthens the guard's own message to name the offending `quant_method`
  type.

In `speculator.py` it is SIX things and none of them is infrastructure. Every one
lands in W4's scope, and they are enumerated here because a W4 implementer who
read the old sentence would under-scope the wave:

- `_selector_walk_kernel` gains a `SAMPLE_PROBABILISTIC: tl.constexpr`, and the
  caller passes `self.draft_logits is not None` for it.
- the kernel's hand-written greedy/Gumbel branch is DELETED and replaced by one
  call to `gumbel_noised_argmax` (`vllm.v1.worker.gpu.sample.gumbel`), with
  `temperature if SAMPLE_PROBABILISTIC else 0.0`. `tl_rand32`, `tl_rand64` and
  `tldevice` are no longer imported.
- the scores are loaded at the width the argmax reduces in
  (`tl.float64 if USE_FP64 else tl.float32`) rather than always `tl.float32`,
  which the head's own comment attributes to a Triton type mismatch on ROCm.
- `self._selector_tokens` is DELETED. The walk writes `self.draft_tokens`
  directly, and the trailing `copy_` is gone with it.
- `self.draft_logits` is no longer allocated in `__init__`; it is `None` for a
  greedy request, is re-filled with `-inf` when it exists, and a new
  `draft_logits_spec(vllm_config)` declares `(torch.float32, -float("inf"))`.
  The head's comment records WHY fp32 and not the head dtype: rounding real
  selector scores to bf16 moves a candidate row's argmax 0.81% of the time and
  reverses 0.68% of candidate pairs.
- `_generate_draft`'s tail is conditional: `_cache_draft_logits` runs only when
  `self.draft_logits is not None`.
- (a SEVENTH, in the BASE class rather than in `dflash2/`, and this section did
  not list it until W4 read the file: `DraftModelSpeculator.__init__` no longer
  hard-codes `torch.zeros(..., dtype=head_dtype)` for the cache. It calls the new
  virtual `draft_logits_spec(vllm_config)` and `torch.full`s with what that
  returns, defaulting to `(head_dtype, 0.0)` with a docstring saying a speculator
  that writes only a SUBSET of columns has to override it. That is the seam the
  DFlash2 override exists to use, and the allocation is still guarded by
  `draft_sample_method == "probabilistic"` at both heads.)

**W4 VERIFIED THIS ENUMERATION rather than trusting it.** All six items above
were re-read at `66e5414c6d75a8529473d977f7458c140bbab8a0` in a local clone and
diffed against `19c93519` with `git diff`; every one is present as described.
The seventh is the only correction, and the `+24 / −6` row it belongs to was
missing from the table.

`_score_edges`, `CandidateSelector`, `hidden_projection`, the two codebooks,
`_topk`, `output_multiplier` and `final_logit_softcapping` are BYTE-IDENTICAL at
the two heads: diffing `qwen3_dflash2.py` for each of those names between them
returns nothing. So the SELECTOR's math is unaffected by the move — which is the
statement W3 needed and the one that was over-generalised into a claim about the
speculator.

This is recorded because "the anchors moved" and "the port must change" are
different statements, and which one is true depends on WHICH FILE — the
distinction the old wording lost.
Both sit past our parity pin `555967922`, which does not carry the architecture
at all, so every anchor below is BEYOND-PIN and cites the PR head rather than the
pin. `## Risks/decisions` D1 argues the posture; the precedent is
`SPEC-DSPARK-QWEN3-ROUTING` toward vllm#52197
([#1193](https://github.com/mudler/vllm.cpp/issues/1193)).

Anchors at that head:

| Upstream | What |
|---|---|
| `vllm/model_executor/models/qwen3_dflash2.py:1-346` | the conv, the selector, `DFlash2Qwen3{DecoderLayer,Model,ForCausalLM}` |
| `vllm/model_executor/models/qwen3_dflash.py` (`_dflash_layer_causal`) | `is_causal` resolved BEFORE the legacy fallback; `decoder_layer_cls`/`model_cls` subclass seams |
| `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:1-224` | `DFlash2Speculator`, the walk kernel, the draft-logit cache |
| `vllm/model_executor/models/registry.py` | `"DFlash2DraftModel" -> ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM")` |
| `vllm/config/vllm.py` (`use_v2_model_runner`, `_is_dflash2_draft`) | a DFlash2 draft forces the V2 runner |
| `vllm/v1/worker/gpu/spec_decode/__init__.py` | speculator selection by draft architecture |

The two mechanisms, in upstream's own terms:

1. **Grouped dynamic depthwise convolution**, wrapping each attention and each
   MLP sublayer — `prepare` before, `finish` after —
   `out[i,c] = sum_t (base[t,c] + delta[i,t,g(c)]) * x[i-t,c]`, taps zeroed
   across the block boundary so a proposal position sees the ones before it
   without another backbone pass. Both coefficient sets come from ONE projection
   of the sublayer input, split into a prepare half and a finish half.
2. **Candidate selector**, replacing the independent per-slot argmax: keep the
   target head's top-K per slot, score adjacent transitions
   `edge(p->c) = <A[p] * project(h), B[c]> + unary[c]`, walk the best path from
   the verified anchor. At T>0 the walk draws with GUMBEL-MAX noise keyed by the
   candidate token ids and caches the realized scores as the proposal
   distribution q over the K candidates, which the lossless verify requires.

   **"Inverse CDF" is what this line said until W4, and it was wrong at BOTH
   heads** ([#1501](https://github.com/mudler/vllm.cpp/issues/1501)). At `19c93519` the walk's non-greedy branch already drew
   `-log(-log1p(-u))` Gumbel noise and took the argmax of `scores/temperature +
   noise`; at `66e5414c` that hand-written branch is replaced by a call to
   `gumbel_noised_argmax`, which is the same draw. No inverse-CDF walk exists
   upstream. The mistake was inherited from the brief that opened this row and
   never re-read against the source, which is why `## Gates` requires the port to
   cite what it read. It changes nothing that landed -- W4 ships the GREEDY arm
   and `## Risks/decisions` D13 records why the noised arm is not reachable here
   -- but it would have mis-scoped the wave that lands it.

Upstream's measurements, recorded as THEIR numbers and not as an expectation for
this engine (H200, `Qwen/Qwen3.8-27B`, k=7): per-request acceptance 5.34 against
DSpark's 4.27; throughput 224.6 against 178.5 tok/s at concurrency 1, 2759
against 2221 at 32; conv plus selector 0.67-0.84% of the serving step.

**The checkpoint is the authority on shapes.** `z-lab/Qwen3.8-27B-DFlash2` is
public; its safetensors header was range-read on 2026-08-19 (81 tensors) and is
the DFlash1 set plus exactly:

```
layers.N.attention_conv.base_kernel               bf16 (2, 2, 5120)   x5
layers.N.attention_conv.kernel_projection.weight  bf16 (1280, 5120)   x5
layers.N.mlp_conv.base_kernel                     bf16 (2, 2, 5120)   x5
layers.N.mlp_conv.kernel_projection.weight        bf16 (1280, 5120)   x5
candidate_selector.hidden_projection.weight       bf16 (256, 5120)
candidate_selector.predecessor_codebook           bf16 (248320, 256)   127 MB
candidate_selector.successor_codebook             bf16 (248320, 256)   127 MB
```

`config.json`: `conv_kernel_size 2` (taps), `conv_group_size 16`,
`selector_rank 256`, `selector_top_k 16`, `block_size 8`, `is_causal false`, 5
layers, hidden 5120, vocab 248320, `target_layer_ids [5,19,33,47,61]`. Derived
and checked against the header: `num_groups = 5120/16 = 320`, so
`kernel_projection` out is `2*taps*num_groups = 1280`, and `base_kernel` dim 0 is
the prepare/finish side rather than a tap.

Three scalars upstream reads with a default — `input_embedding_scale`,
`output_multiplier`, `final_logit_softcapping` — are ABSENT from the Qwen3.8
draft's config, which takes 1.0, 1.0 and disabled.

**`z-lab/Muse-Glimmer-30B-DFlash2` sets two of them**, so they are
CHECKPOINT-EXERCISED rather than synthetic
([#1327](https://github.com/mudler/vllm.cpp/issues/1327)). Its `config.json`
(sha256 `cb684d6f688a22619a63ea1debe7d30c139c195bf3141fd86a763763ab34b5d9`, read
2026-08-19) declares `output_multiplier 0.19611613513818404` and
`final_logit_softcapping 20.0`, together with `block_size` 16, hidden 6656 (so
`num_groups` 416 and a 1664-wide `kernel_projection`), `rope_theta` 500000.0
nested under `rope_parameters`, and vocab 202048. Both scalars are applied to the
candidate VALUES in `compute_candidates` BEFORE the selector scores them
(`qwen3_dflash2.py` `DFlash2Qwen3ForCausalLM.compute_candidates` @ the PR head),
so a wrong value reorders the top-K and moves acceptance without raising — the
`is_causal` failure mode one layer up. This is the same pair Muse Glimmer's text
tower already needed here: `docs/USAGE.md` records that the released 30B config
carries both while the GGUF and the DFlash drafter each omit some, and that both
used to run a quietly different model. `input_embedding_scale` remains
unexercised by both published drafts and stays synthetic.

An earlier revision of this section said no published checkpoint exercised any of
the three. That was measured on the Qwen3.8 draft alone.

## Our baseline

Present, and reused rather than rebuilt:

- The whole DFlash lane is DONE and merged: `src/vllm/model_executor/models/qwen3_dflash.cpp`
  (1160 lines), `qwen3_dflash_weights.cpp`, `qwen3_dflash_gguf.cpp`,
  `src/vllm/v1/worker/gpu/spec_decode/dflash/speculator.cpp`, with the runner
  loop, the rejection path and the GDN-slot rollback shared with MTP.
- `vt::DFlashBlockAttention` (`KERNEL-ATTN-DFLASH-BLOCK`) is the non-causal
  in-block primitive DFlash2 needs unchanged, including the warp-scoped paged
  block kernel that closed the `SPEC-DFLASH` speed gate.
- The loader ALREADY shares `embed_tokens` and `lm_head` from the target, which
  is exactly what a DFlash2 checkpoint requires
  (the `TryLoadBf16` comment on `src/vllm/model_executor/models/qwen3_dflash_weights.cpp` and the two calls it
  documents in `src/vllm/model_executor/models/qwen3_dflash_weights.cpp::LoadQwen3DFlash`; cited by symbol because the range moved
  three times inside this one branch as comments above it grew). The
  class comment on `include/vllm/model_executor/models/qwen3_dflash.h`'s
  `Qwen3DFlashWeights` claims the draft owns both; it is STALE and this row
  corrects it. (Cited by symbol: the line range first written here, `:79-82`,
  named four fields of `Qwen3DFlashLayerWeights` instead.)
- `src/vt/cuda/cuda_sample.cu:297-506` is a sort-free block-cooperative
  pivot-bracket threshold search, one block per row, ported from the same
  FlashInfer `TopK/TopPRenormProb` approach the selector's top-k uses.
- `Qwen3DSparkModel::SampleSequentialDevice` is the shipped precedent for a
  sequential per-step draft walk that runs on device rather than on the host.

Absent, and owed by this row:

- No route for `DFlash2DraftModel`. `SpeculativeConfig` classified a DSpark draft
  from its own `config.json` (`include/vllm/config/speculative.h:159-182` after
  W1's insertion above it) and had no DFlash equivalent for a second architecture.
  CLOSED by W1: `SpeculativeConfig::IsDflash2Draft`
  (`include/vllm/config/speculative.h:115-138`) with its loader refusal
  (the classification helper
  `src/vllm/entrypoints/model_loader.cpp::ReadDflashDraftArchitectures` and
  `src/vllm/entrypoints/model_loader.cpp::CheckDflash2DraftArm` (named `RefuseDflash2Draft` until W2 split the two container arms), cited by SYMBOL
  because the line range this spec first carried was stale two merges later and
  `scripts/check-symbol-anchors.py` states it cannot verify a line citation).
  The identical gap for
  `DSparkDraftModel` is open as [#1193](https://github.com/mudler/vllm.cpp/issues/1193);
  this row does not fix that one, and must not collide with it.
- No grouped dynamic convolution anywhere in `vt`.
- No candidate selector, no lattice, no path walk.
- No top-k that EMITS the surviving (id, value) pairs. The threshold search
  above masks below the k-th largest and returns no indices.
- `is_causal` was not read: causality resolved by the legacy rule alone. CLOSED by
  W1 in `src/vllm/model_executor/models/qwen3_dflash_weights.cpp::ResolveQwen3DFlashAttnModes`,
  with its coercion helper `src/vllm/model_executor/models/qwen3_dflash_weights.cpp::DeclaredCausal`, and
  `src/vllm/model_executor/models/qwen3_dflash_weights.cpp::MakeQwen3DFlashDraftConfig`.

## Port map

| Upstream | Ours | Note |
|---|---|---|
| `qwen3_dflash2.py` conv | `vt::DFlashGroupedConv` op, kernel-matrix row `KERNEL-DFLASH2-GROUPED-CONV`, wrapped into the existing `src/vllm/model_executor/models/qwen3_dflash.cpp` layer bodies | CPU reference first, CUDA after, as `KERNEL-ATTN-DFLASH-BLOCK` did. LANDED in W2. The conv wraps sublayers of the SHIPPED DFlash block rather than a new file, because upstream subclasses `DFlashQwen3DecoderLayer` and overrides only `forward`; a parallel `qwen3_dflash2.cpp` copy of that body is what AGENTS.md `## Shared seams` forbids |
| `qwen3_dflash2.py` selector | `src/vllm/model_executor/models/qwen3_dflash2.cpp` + `vt::Dflash2SelectorEdges`, kernel-matrix row `KERNEL-DFLASH2-SELECTOR-EDGES` | LANDED in W3. `_score_edges` is one einsum; it is not the cost. The op is NOT bit-identical across backends -- the rank contraction is a reduction -- unlike `KERNEL-DFLASH2-GROUPED-CONV`, which is elementwise |
| `_topk` / FlashInfer radix | `vt::TopKValuesIndices`, kernel-matrix row `KERNEL-TOPK-PAIRS`, extending the pivot-bracket search in `src/vt/cuda/cuda_sample.cu` to emit pairs | D2. LANDED in W3. The TIE-BREAK is the contract: descending value, ties by ascending index |
| `dflash2/speculator.py` walk kernel | `vt::Dflash2PathWalk`, kernel-matrix row `KERNEL-DFLASH2-PATH-WALK`, consumed by `src/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.cpp::Dflash2WalkPath` | D3: device from day one. LANDED in W4, in upstream's OWN grid -- one program per request with the step loop inside it. Specified BIT-EXACT across backends, unlike the lattice op: the walk compares and gathers and performs no arithmetic |
| `dflash2/speculator.py` draft-logit cache | not ported; `## Owed` O12 | D13. At the MOVED head `_cache_draft_logits` runs only `if self.draft_logits is not None`, which is set only for `draft_sample_method == "probabilistic"` -- a value `ParseSpeculativeConfigJson` refuses BY NAME here, against a verify that is accept-iff-equal. Its layout is recorded (`draft_logits_spec`: fp32, `-inf` fill, `None` for greedy) so the wave that lands the probabilistic arm does not re-derive it |
| `registry.py` + `spec_decode/__init__.py` | `include/vllm/config/speculative.h`, `src/vllm/entrypoints/model_loader.cpp` | classify from the draft's own config, as `IsDsparkDraft` does |
| `_dflash_layer_causal` | `src/vllm/model_executor/models/qwen3_dflash_weights.cpp::DeclaredCausal` + `src/vllm/model_executor/models/qwen3_dflash_weights.cpp::ResolveQwen3DFlashAttnModes` | D4; the ONE shared-behaviour edit. LANDED in W1 |
| `use_v2_model_runner` | no analogue | we have one runner; record the reason rather than porting a switch |
| — | `src/vllm/model_executor/models/qwen3_dflash2_gguf.cpp` | the GGUF drafter arm, W5 |

New files mirror the upstream file structure, as AGENTS.md requires. Nothing is
added to `qwen3_dflash.cpp` except the causality resolution.

## Tests to port

From the PR head, parameters and tolerances preserved:

- `tests/v1/spec_decode/test_dflash2.py::test_grouped_conv_matches_reference`,
  both `block_size` parameters (5 and 8 — 8 exercises the power-of-two masking
  arm, 5 the modulo arm).
- `tests/v1/spec_decode/test_dflash2.py::test_selector_edges_match_sequential_reference`.
- `tests/test_config.py::test_dflash2_draft_forces_v2_model_runner`, adapted:
  our engine has one runner, so the ported assertion is that a DFlash2 draft
  reaches the DFlash2 speculator and never the DFlash1 one.
- `tests/v1/spec_decode/test_dflash_causality.py` as edited by the PR, which is
  what covers D4.
- `tests/models/registry.py` DFlash2 entry, as the checkpoint pin.

Ours, red-first, beyond the ports:

- A causality gate that FAILS on the legacy rule against the real
  `config.json` shape (all layers `sliding_attention`, `is_causal false`). The
  mutation is restoring the old resolution; the gate must redden.
- A reachability gate per `.agents/reachability.md`: delete the production call
  site of the selector in a scratch copy and the focused gate must redden. A
  unit test that constructs the selector by hand proves the class works and not
  that any draft reaches it.
- A lower-bound gate on the GGUF arm, per the standing rule that a token gate
  cannot see a dequantizing fallback.

## Gates

**Correctness, before any speed number.**

- G1: conv and lattice against upstream's references, CPU then CUDA, at the
  checkpoints' real shapes (taps 2, group 16, K 16, rank 256).
  **G1 runs at BOTH published block shapes, 8 and 16.** Upstream's own reference
  test parametrises `block_size` 5 and 8 to cover the power-of-two branch of the
  position mask (`pos & (block-1)` against `pos % block`); 16 is the shape a real
  checkpoint ships (`z-lab/Muse-Glimmer-30B-DFlash2`) and neither upstream
  parameter reaches it ([#1327](https://github.com/mudler/vllm.cpp/issues/1327)).
  W2 discharges the conv half of G1 on CPU at 5, 8 and 16; the CUDA half is
  written and registered but UNVERIFIED (no `nvcc` on the authoring host) and is
  owed to a GPU lease.
- G2: draft-token identity against vLLM at PR head `66e5414c` on
  `z-lab/Qwen3.8-27B-DFlash2` over `Qwen/Qwen3.8-27B`, identical prompts and
  identical k, greedy. **RECONCILED TO ONE HEAD BY W6 on 2026-08-21**, to
  `66e5414c`. It read `19c93519` until then, and the reconciliation is applied in
  place rather than annotated, because a gate head and a port head that differ by
  name are exactly what the paragraph below warns against.

  **W6 APPLIED THIS RULE'S WRONG BRANCH, AND THE RESULT STANDS ANYWAY BECAUSE
  THE CAPTURE PREDATES THE MERGE.** The rule below is "`66e5414c` if #52816 has
  not merged, and the merge commit if it has". #52816 HAD merged --
  `2026-08-21T05:27:22Z`, merge commit `b389ac29`, head `3406ec1d` -- 46 minutes
  before W6's work commit was authored, and W6 recorded it as open without
  re-reading the forge. What the rule selects today is therefore `b389ac29` and
  not `66e5414c`.

  **The gate head stays `66e5414c` for the W6 capture, and this is a DATED
  exception rather than the rule.** The reason is that the capture was taken
  against a wheel built at `66e5414c` on a leased GPU BEFORE the merge existed
  to select; re-labelling that run with a head it never executed would be a
  false pin, and the honest move is to date what was measured. It is not a
  reason to leave the head at `66e5414c` for the NEXT reading of these gates:
  [#1561](https://github.com/mudler/vllm.cpp/issues/1561) owns moving the gate
  head to `b389ac29` and re-reading G2 and G3 there, and until that happens this
  row's gates are read against a head that is one merge behind vLLM's `main`.
  `## Owed` O21 carries the difference between the two heads, re-measured. The DFlash near-tie envelope applies: `SPEC-DFLASH`
  established that strict token identity is bf16-irreducible on portable
  kernels, so the ratified near-tie gate is the admissible form and a strict
  claim is not.
  **`19c93519` here is a CHOICE and not a leftover, and W6 owns the final
  call.** The port mirrors `66e5414c`, which superseded `19c93519` on 2026-08-19
  ([#1404](https://github.com/mudler/vllm.cpp/issues/1404)), so the gate head and
  the port head differ by name. It is defensible because the greedy ANSWER is
  identical at both: the two heads differ in this path only by collapsing the
  walk's hand-written `temperature == 0.0` and Gumbel branches into one
  `gumbel_noised_argmax` call, by deleting the private `_selector_tokens` buffer
  and its `copy_`, and by making the proposal distribution optional and `None`
  for greedy — none of which a greedy run can observe, and `## Upstream chain`
  carries the diff. It is stated rather than assumed because a gate head that
  merely drifted from the port head is how a parity claim quietly stops meaning
  what it says. W6 takes the gate and reconciles this to ONE head at that
  point: `66e5414c` if #52816 has not merged, and the merge commit if it has.
- G3: **acceptance**, measured SAME-TRAJECTORY. `SPEC-DFLASH` D8 spent a whole
  campaign on an acceptance deficit that was a divergent-trajectory measurement
  confound, and D9 refuted it. This gate does not repeat that mistake.

  **NOTHING IS TEACHER-FORCED, and this line said it was until 2026-08-21.**
  G2's wording was reconciled to the implementation during W6 and G3's was not,
  so the spec kept describing a mechanism the gate has never had. What the gate
  actually does is an ADMISSION CONDITION: both engines run FREE, and a prompt
  contributes to the acceptance comparison only when the two independently
  emitted the SAME token stream
  (`tests/parity/test_qwen38_dflash2_spec_decode.cpp:640-645`, the `is_exact`
  admission). A prompt that
  diverges is excluded and named.

  The substitution is defensible and is arguably the stronger claim -- a
  same-trajectory pair that arose without being forced is evidence about the
  engine, where a forced one is evidence about the verify rule alone -- but it
  costs something the forced form would not, and the cost is stated rather than
  hidden. **G3 is UN-TAKEABLE on a diverging prompt.** With no same-trajectory
  prompt the gate reports "G3 NOT TAKEN", which it correctly calls a measurement
  gap rather than an acceptance result. That is exactly the regime the D8/D9
  confound arose in, so on the day this row has to answer for a divergence, G3
  will have nothing to say until an actual teacher-forced arm is built.
- G4: the GGUF drafter arm, with the lower bound of `## Tests to port`.
  **DISCHARGED by W5 on CPU**, at the three published encodings (bf16, Q8_0,
  Q4_K) and with the lower bound's three legs measured rather than argued:
  the block-encoded byte count, bit-exact values against the suite's own
  encoders, and difference from the bf16 arm at both the tensor and the block
  logits. What G4 does NOT cover and G2/G3 still owe is a GGUF-sourced DFlash2
  draft running against a real 27B target on a device.
- G5: reachability, as above. **DISCHARGED by W4** for the whole DFlash2 chain:
  `tests/vllm/v1/spec_decode/test_dflash2_runner_reach.cpp` drives a real
  `LoadedEngine` through `GPUModelRunner::propose_drafts_block`, the engine
  GENERATES, and the drafts it emits are read off the production trace and
  required to move with the selector's own output scalars — which the DFlash1
  per-slot argmax cannot do. Deleting the runner's walk call site reddens it, and
  so does replacing the walk with that argmax. `## Now` carries the full mutation
  set and the two that came back green.

**Speed.** No ratio is claimed by this row until G2 and G3 read. When one is
taken it uses vLLM's production configuration as the denominator, never
`--enforce-eager`, on an idle leased host with a same-binary A/B, per
`.agents/benchmarking.md`. Upstream's H200 numbers are not a floor for this
engine and are not treated as one.

**Oracle. RECONCILED BY W6 ON 2026-08-21, AND IT IS BUILT AND IT RUNS.** The
pinned oracle cannot run this architecture. The gate oracle is vLLM at
`66e5414c6d75a8529473d977f7458c140bbab8a0` — the head the port mirrors — and it
is a BEYOND-PIN oracle rather than a pin advance. This section named `19c93519`
until W6.

**W6 wrote that #52816 "was still OPEN when W6 applied it (read from the forge,
not assumed)", and that is the one claim in this section that is false.** The
pull request had merged 46 minutes earlier. Whatever W6 read, it was not the
state of the forge at the time it committed. Corrected 2026-08-21 by the repair
wave, which re-read it: merged `05:27:22Z`, merge commit `b389ac29`, head
`3406ec1d`.

The oracle stays `66e5414c` because that is the wheel that was BUILT AND RUN on
the lease. `66e5414c` is an earlier head of #52816 than the merged `3406ec1d`,
but nothing in the tree dates the build or the run, so where that run falls
against the merge instant is unmeasured -- and it does no work either way, since
what keeps the head is that it is what executed. The `#1561` index row states
the same thing as "it predates the merge", which is that unmeasured ordering; it
is append-only and is not edited, and this paragraph is the authority.
`## Gates` G2 carries why the pin is a dated exception and not the rule, and
[#1561](https://github.com/mudler/vllm.cpp/issues/1561) owns moving it.

The artifact, so a later reader can tell whether they hold the same one:

| | |
|---|---|
| wheel | `vllm-0.1.dev1+g66e5414c6-cp312-cp312-linux_aarch64.whl` |
| sha256 | `fbc247ab1bda93a81ff7a68658cdda65b697e263ad2c43a2bc62c2591d207439` |
| runtime version | `0.1.dev1+g66e5414c6`, asserted from `/` so the source tree cannot answer |
| `DFlash2DraftModel` | registered, asserted against `_SPECULATIVE_DECODING_MODELS` |
| gateable | **yes** — it loads `Qwen/Qwen3.8-27B`, captures CUDA graphs including the DFlash2 speculator's own, and GENERATES with speculation live |

**Three operational facts about driving it, each of which cost a 51.75 GiB
load.** They are here rather than in a runbook because the next agent to
instrument this oracle needs them before writing a script, not after.

1. **`VLLM_ATTENTION_BACKEND` does not exist at this revision.** Grepping every
   `.py` in the wheel returns nothing. The knob is `EngineArgs.attention_backend`
   (`arg_utils.py:706`), folded into `AttentionConfig.backend` (`:2382`) and read
   in `vllm/v1/attention/selector.py:179`. The old export selects nothing and
   auto-selection wins SILENTLY, so a run can record one backend while executing
   another. Pass the kwarg and read the resolved value back off the built engine.
   `## Owed` O22, [#1456](https://github.com/mudler/vllm.cpp/issues/1456).
2. **The engine core is a separate process by default**, so an in-process
   monkeypatch never reaches the object that drafts.
   `VLLM_ENABLE_V1_MULTIPROCESSING=0` is the switch. `## Owed` O23.
3. **`capture_model()` calls `_generate_draft`**, so any host copy in a hook on
   it is illegal inside a CUDA graph capture. `## Owed` O23.

**And the constraint this row inherited is weaker than it was written.** `## Gates`
declares `TRITON_ATTN` because #1456 concluded the GB10 oracle has no
`FLASH_ATTN` denominator; a W6 run generated on `FLASH_ATTN` anyway. W6 keeps the
DECLARED backend rather than substituting one, and takes both arms.
`## Owed` O22 carries it.

## Dependencies

- A GPU lease on a fleet device (`rc run`/`rc hold`) for every G2-G5 run. Never
  `ssh` to a fleet box.
- The 27B target and the DFlash2 drafter resident where the gate host can read
  them, with a recorded revision, since a repo id alone is not a pin.
- A vLLM build at `66e5414c`. The parity pin stays where it is. **W6 SETTLED
  this on 2026-08-21 and the build EXISTS**: `## Oracle` carries the wheel, its
  sha256, its asserted runtime version and the three operational facts about
  driving it. The line read `19c93519` through W5 — the head #52816 was opened
  at, superseded on 2026-08-19
  ([#1404](https://github.com/mudler/vllm.cpp/issues/1404)) — and is reconciled
  in place rather than annotated, for the reason G2 gives.
- [#1193](https://github.com/mudler/vllm.cpp/issues/1193) touches the same
  classification code. Whichever lands second reconciles; neither blocks.
- No dependency on the V1/V2 runner distinction, which this engine does not have.

## Work breakdown

Each wave is landable alone, red-first, with its own focused gate and a fresh
reviewer who mutates the guarantee rather than reading it.

- **W1 — routing and the causality rule.** Classify `DFlash2DraftModel` from the
  draft config; refuse an unimplemented arm BY NAME rather than degrading to
  DFlash1. Land D4's `is_causal` precedence with its red-first gate. Smallest
  wave, and the one that removes the silent-wrong path.
- **W2 — the grouped convolution.** CPU reference, then CUDA, then the draft
  block wiring. G1's conv half.
- **W3 — the selector.** Lattice op, codebooks in the loader, the top-k that
  emits pairs (D2). G1's lattice half.
- **W4 — the speculator. LANDED 2026-08-20.** The device path walk and the
  wiring that makes a DFlash2 draft reach it; G5 landed here. The head-move
  warning this entry carried was correct and it bit exactly where it predicted:
  the T>0 arm this wave was scoped to write **is not inverse CDF at the moved
  head** — it is Gumbel-max through `gumbel_noised_argmax` — and it is not
  landed at all, because no configuration this engine admits can reach it. D13
  carries that decision and `## Owed` O12 carries what it leaves owed. The
  realized-q draft-logit cache moved with it, for the same reason and by
  upstream's own polarity: at `66e5414c` `_generate_draft` calls it only when a
  proposal distribution exists.
- **W5 — the GGUF drafter arm**, with its lower bound. LANDED 2026-08-20.
- **W6 — the gates.** G2 and G3 on a leased GPU against the PR-head oracle,
  then `## Outcome`.
- **W7 — async scheduling for the Eagle-type speculative family
  ([#1824](https://github.com/mudler/vllm.cpp/issues/1824)). LANDED
  2026-08-23.** The engine forced synchronous scheduling under ANY speculator
  (a SPEC-MTP I5d deferral); upstream keeps async ON for Eagle-type methods,
  dflash included, and at c1 that difference serializes every host-side
  scheduling cost into each of ~360 steps — the largest named host-side
  divergence in the #1574 gap. W7 ports the draft-in-output flow and flips the
  enable to upstream's method predicate. Own spec:
  [spec-decode-async-scheduling.md](spec-decode-async-scheduling.md); the c1
  TPOT A/B (async-ON vs `VT_ASYNC_SCHED=0`, same binary, #1574 workload) is
  owed there as A1, operator-run under an `rc` lease.

  **The gate head is reconciled to ONE head here and it is `66e5414c`, which is
  NOT what G2's rule selects.** W6 wrote that vllm#52816 was "still OPEN on
  2026-08-21 (checked through the forge, not inferred)". It was not: it merged
  at `05:27:22Z` that morning, 46 minutes before this wave's work commit, at
  head `3406ec1d` and merge commit `b389ac29`. G2's rule therefore selects
  `b389ac29`, and W6 selected `66e5414c`.

  The measurement is not invalidated by that, because the capture ran against a
  wheel built at `66e5414c` BEFORE the merge landed, and a run cannot be
  re-pinned to a head it never executed. What is invalidated is the REASON W6
  gave. The head is kept, dated, and the reconciliation onto merged upstream is
  [#1561](https://github.com/mudler/vllm.cpp/issues/1561). `## Owed` O21 carries
  the two-head diff, re-measured against the merged head.

  **THE ORACLE'S DRAFT TOKENS COME FROM A HOOK, AND THE HOOK'S OWN PRECONDITIONS
  ARE THE HARD PART.** vLLM exposes no per-block draft-token counter, so G2's
  distinctive half — draft identity rather than output identity — needs
  `DFlash2Speculator._generate_draft` instrumented. Two things make that hook
  silently dead, and both cost a full 51.75 GiB load to discover: the engine core
  runs in a SEPARATE PROCESS by default, and `capture_model()` calls the hooked
  method inside a CUDA graph capture where a host copy is illegal. `## Owed` O23
  carries both with their exact failure text. The capture ABORTS on zero blocks
  rather than writing an empty golden, which is what turned the first failure
  into a message instead of a wrong measurement.

  **The oracle's per-block ACCEPTANCE is reconstructed, not counted**, because
  vLLM's counter is pooled over the run. The verify is accept-iff-equal under
  greedy, so a block's accepted count is the longest prefix of its draft that the
  output took. That derivation is validated twice before any number is read from
  it: against OUR production `[SPECTRACE]` line, which prints the true per-block
  count, and against vLLM's own aggregate counter.

## Risks/decisions

- **D1 — mirror the unmerged PR now, rather than waiting for it to merge.**
  Developer decision, 2026-08-19. The mechanism is architecturally separate from
  DFlash1, and two open pull requests whose second one only fixes a guard suggest
  a settled design. Cost: the anchors can move under review, and the port
  reconciles if they do. Precedent: `SPEC-DSPARK-QWEN3-ROUTING` toward
  vllm#52197. This does NOT advance the parity pin.

  **SETTLED 2026-08-21: it merged.** vllm#52816 merged at `05:27:22Z` at head
  `3406ec1d`, merge commit `b389ac29`, so the bet D1 took paid off and the
  anchors did move under review exactly as the cost line predicted — three times.
  The port is NOT yet reconciled onto merged upstream and that is
  [#1561](https://github.com/mudler/vllm.cpp/issues/1561). The parity pin is
  still not advanced by this row.
- **D2 — do not port FlashInfer's radix top-k.** `topk.cuh` is 3380 lines of
  general kernel: multi-CTA, deterministic mode, three tie-break modes, dynamic
  shared-memory sizing. Our shape is fixed and small — K=16 over 248320 for
  `num_reqs * k` rows, about 224 at concurrency 32 — and
  `src/vt/cuda/cuda_sample.cu:297-506` already carries the same pivot-bracket
  threshold search, gated. Extending it to compact the survivors and order at
  most K of them is the smaller and better-covered change. Revisit only if W3
  measures the top-k as the selector's dominant cost here, as it is upstream.
- **D3 — the path walk runs on device from the first landing.** The host-side
  arm is not an acceptable first version: the identical shape in DSpark measured
  28% of the 27B draft step ([#436](https://github.com/mudler/vllm.cpp/issues/436))
  and had to be moved. A host arm may exist as an A/B lever behind an
  environment switch, as `VT_DSPARK_DEVICE_SAMPLE` does, and not as the default.
- **D4 — `is_causal` takes precedence over the legacy layer rule.** This is the
  one edit that touches shipped DFlash behaviour, and it is upstream's own
  change in the same commit. Without it the published DFlash2 checkpoint — all
  five layers `sliding_attention`, `is_causal false` — runs every layer CAUSAL,
  emits plausible tokens, passes a token gate against our own output, and only
  loses acceptance. It is the single defect in this row that raises nothing.
  Existing DFlash checkpoints carry no `is_causal`, so their resolution is
  unchanged; the gate asserts that too.
- **D5 — the codebooks cost ~254 MB resident bf16** that the DFlash1 lane never
  allocates, at `selector_rank 256` and a 248320 vocabulary. This is recorded as
  a memory axis of the row rather than discovered on a gate host. Whether they
  can be held in a narrower dtype is NOT decided here; upstream stores bf16 and
  we mirror upstream.
- **D6 — the GGUF arm lands in wave 1 of the campaign**, not as a follow-on row.
  Developer scope answer, 2026-08-19. `SPEC-DFLASH` split its GGUF arm into
  `SPEC-DFLASH-GGUF`, which reached `DONE` separately; this row carries the arm
  itself rather than repeating the split.
- **D7 — no ceiling is declared anywhere in this spec.** If our acceptance or
  throughput reads below vLLM's on the same workload, that is an unresolved
  implementation difference with a named next hypothesis, per AGENTS.md.
- **D8 — DIVERGENCE: an uncoercible `is_causal` is REFUSED by name, where
  upstream coerces it.** Upstream resolves the key as `bool(is_causal)`
  (`qwen3_dflash.py:58-67` @ vllm-project/vllm#52816 head
  `19c9351904df4c63042671bc67a866ca48dc7d6f`), and Python's `bool` accepts every
  object: `bool("false")` is `True`, `bool([])` is `False`. `DeclaredCausal`
  (`src/vllm/model_executor/models/qwen3_dflash_weights.cpp::DeclaredCausal`)
  instead honours a boolean or a number and refuses any other JSON type with a
  message naming the key. This is one exact tracked exception under AGENTS.md
  `## vLLM is the reference`, and the reason is the OTHER container rather than
  Python. The GGUF spelling `dflash.attention.causal` is read through `KvI64`,
  which takes every integer width and bool and names its own error on anything
  else, so a `"false"` string has no GGUF counterpart to agree with: coercing it
  in the HF arm would make the two containers answer differently for the same
  logical checkpoint, which is the exact failure `## Now` records for the numeric
  spelling one step earlier. The blast radius is empty on every published
  artifact -- all four DFlash draft `config.json` files read on 2026-08-19
  (`z-lab/Qwen3.6-27B-DFlash`, `z-lab/Qwen3.5-9B-DFlash`,
  `z-lab/gemma-4-31B-it-DFlash`, `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash`) declare
  no `is_causal` at all, and `z-lab/Qwen3.8-27B-DFlash2` declares it as a JSON
  boolean. Reconcile onto upstream's coercion if a checkpoint ever spells it as a
  string, and change the GGUF arm in the same edit or not at all.
- **D11 — `set_model_tag` is a DELIBERATE NON-PORT.** W3 decision, 2026-08-20,
  from the moved head ([#1404](https://github.com/mudler/vllm.cpp/issues/1404)).
  Upstream wraps the selector's construction in
  `set_model_tag("dflash2_candidate_selector")` because `CandidateSelector`
  carries its own `@support_torch_compile` and is built while the DRAFT's model
  tag is still active, so without a tag of its own the two share one
  compile-cache namespace and the selector loads the draft's graph — a different
  input signature, within the same startup. It is the ONLY behavioural change
  #52816 made to that file between its two heads. This engine has no
  torch.compile and no compile cache, so there is nothing for a tag to
  disambiguate and nothing to port. Recorded here, and on
  `Dflash2SelectorWeights::kNonPortSetModelTag`, rather than skipped in silence:
  a later reader diffing our selector against upstream's will find the wrapper
  missing and needs to know it was decided.
- **D12 — the WIDE LM-head guard is ported, and it is LIVE here.** W3 decision,
  2026-08-20. Upstream refuses a quantized target LM head by name in
  `compute_candidates`, and the new head's guard admits BOTH
  `UnquantizedEmbeddingMethod` and `UnquantizedLinearMethod` — the second is the
  folded-in [#52883](https://github.com/vllm-project/vllm/pull/52883), because a
  `ParallelLMHead` returns the LINEAR method whenever a quant config leaves the
  head itself unquantized (INC, ModelOpt, fp8 with excluded layers), so the
  narrow guard refused valid heads. The wide form is what `RefuseQuantizedDflash2LmHead`
  mirrors: any head readable as dense floats is admitted, whatever loaded it.
  It is not decoration here. **AMENDED by D14 on 2026-08-21
  ([#1628](https://github.com/mudler/vllm.cpp/issues/1628)): the sentence that
  follows was true when it was written and is no longer.** The safetensors arm
  refused a non-BF16 head one
  layer up in `LoadNamedBf16`, but the GGUF arm DEQUANTIZES a q6_K/NVFP4
  `output.weight` to bf16 (`LoadGgufSharedEmbedAndHeadBf16`), and a GGUF target
  with a safetensors DFlash2 draft is an admitted combination today. The
  selector's whole input is the target head's EXACT top-K, so a dequantized head
  produces a different candidate set with no visible symptom. `Qwen3DFlashWeights`
  therefore carries `lm_head_dequantized`, set by the loader from the ggml type
  of the tensor that was read, and the guard reads it. The DFlash1 lane is
  unaffected and that is asserted, because DFlash1 has no selector and has
  shipped the GGUF-target combination since `SPEC-DFLASH-GGUF`.
- **D9 — the row gates the OUTPUT SCALARS against a checkpoint that SETS them,
  not against defaults.** Discovered after this spec landed
  ([#1327](https://github.com/mudler/vllm.cpp/issues/1327)); the brief that
  proposed it called it D8, which was already taken by the `is_causal` coercion
  divergence above, so it is D9 here. A port that reads all three with
  `.get(key, default)` passes every gate built from the Qwen3.8 draft alone,
  because that draft sets none of them — such a gate measures the default path
  and reports it as coverage. W3 owns the scalars and must gate them on
  `z-lab/Muse-Glimmer-30B-DFlash2`'s values.

  **The acceptance-moving half of D9 had a THIN margin, measured by W3's fresh
  review, and it is now wider.** The half that matters is not that the values
  differ — a monotone rescale would do that — but that the ARGMAX over children
  under some predecessor FLIPS, which is what moves the path the W4 walk takes.
  One synthetic block gives `num_steps * K` = 3 * 3 = **9** predecessor slots,
  and exactly **1** of those 9 flipped. `> 0` on 1-of-9 is non-vacuous and one
  arithmetic accident away from passing for the wrong reason, and nothing in the
  case let a reader see how close it was. The case now sweeps five blocks with
  different anchors and different logit and hidden ramps and asserts on the
  aggregate: measured 2026-08-20, **8 flips over 45 slots, in 4 of the 5
  blocks**. The slot count is pinned exactly, because it is a shape; the flip
  counts are floors under the measured values, so a rounding-level change does
  not red the suite while a port that stopped reordering children still cannot
  pass. The zero-reading precondition is asserted per block.
- **D10 — the DFlash2 refusal MOVES from startup to the first propose, on the
  safetensors arm only.** W2 decision, 2026-08-19. W1 refused a
  `DFlash2DraftModel` draft before any weight was read, when both mechanisms were
  missing. W2 implements one of them, and keeping the startup refusal would leave
  every line of it unreachable from any production entry point — AGENTS.md
  `## Nothing lands dead` — with the conv gated only by tests that construct it,
  which is `.agents/reachability.md`'s test-only driver. So a safetensors DFlash2
  draft is now ADMITTED: it loads its conv weights, runs the conv in all three
  layer bodies, and is refused BY NAME after the block forward and before
  anything samples -- at `RefuseDflash2CandidateSelector` as W2 wrote it, and at
  `RefuseDflash2PathWalk` since W3 retired that function and moved the boundary
  one step further out. Cost: the refusal arrives at the
  first generated token rather than at startup. It is paid down by a STARTUP
  NOTICE from `CheckDflash2DraftArm` naming the mechanism that runs, the one that
  does not, the wave that owns it and the issue, so nothing is a surprise. The
  GGUF arm KEPT the startup refusal through W4, because its drafter arm had no
  conv weight path at all and admitting the file would have loaded a DFlash1
  draft out of a DFlash2 checkpoint. **W5 lands that path, so the refusal is gone
  from that container too** and `CheckDflash2DraftArm` now only ever prints. The
  polarity D10 set is what made the sequence safe: each container was admitted in
  the wave that could actually run it, and never one wave earlier.

- **D13 — the PROBABILISTIC draft-sample arm is NOT ported, and the greedy arm
  this row ships is upstream's greedy arm exactly.** W4 decision, 2026-08-20.
  Issues [#1314](https://github.com/mudler/vllm.cpp/issues/1314) and
  [#1501](https://github.com/mudler/vllm.cpp/issues/1501), the latter being the
  record defect this decision was taken while correcting: the row had recorded
  the T>0 walk as inverse CDF since its opening brief, and it is Gumbel-max at
  both heads.

  At the moved head `_selector_walk_kernel` calls `gumbel_noised_argmax` with
  `temperature if SAMPLE_PROBABILISTIC else 0.0`, and `SAMPLE_PROBABILISTIC` is
  `self.draft_logits is not None`, which `DraftModelSpeculator.__init__` sets
  only for `speculative_config.draft_sample_method == "probabilistic"`. At
  temperature 0 that helper divides by nothing, adds no noise, and returns
  `tl.max(logits, axis=0, return_indices=True)`; the `USE_FP64` cast it may apply
  is order-preserving on fp32 inputs and cannot move a greedy answer. So the
  greedy walk this row ports is not an approximation of upstream's walk, it is
  one of its two arms, byte-for-byte.

  **This engine cannot reach the other arm**, and the refusal predates this row
  by a long way: `vllm::ParseSpeculativeConfigJson`
  (`src/vllm/config/speculative.cpp`) refuses `draft_sample_method:
  "probabilistic"` BY NAME, owed to row `SPEC-ACCEPT-VARIANTS`, because this
  engine's verify is accept-iff-equal
  (`include/vllm/v1/spec_decode/rejection_sampler.h`) and there is nothing to
  consume a proposal distribution. Landing the noised arm and the realized-q
  cache would therefore land two mechanisms no production entry point can reach,
  which is what AGENTS.md `## Nothing lands dead` forbids — and it would land
  them WITHOUT a gate, because bit-parity for the noise needs Triton's Philox
  stream (`tl.randint4x`, Philox 4x32-10) reproduced exactly, which is its own
  port with its own oracle.

  What is recorded instead of built is the LAYOUT, so the wave that lands it does
  not re-derive it: `draft_logits_spec` returns `(torch.float32, -inf)` for
  DFlash2 — fp32 rather than the head dtype because rounding real selector scores
  to bf16 moves a candidate row's argmax 0.81% of the time and reverses 0.68% of
  candidate pairs, and an `-inf` fill because `_cache_draft_logits_kernel` writes
  only the K candidate columns and every column it never touches has to read as
  impossible. `## Owed` O12 carries it.

  The half of `draft_logits_spec` that IS honoured here is the one W4 could
  honour: `None` for greedy. This engine allocates no proposal distribution and
  caches no realized scores on the arm it ships, which is the moved head's own
  polarity and the reverse of `19c93519`, where DFlash2 forced the allocation
  unconditionally.

- **D14 — the LM-head guard is NARROWED to the state it argues about, and a
  PACKED target head is computed with rather than refused.** Decision
  2026-08-21, [#1628](https://github.com/mudler/vllm.cpp/issues/1628). D12
  stands and is not reversed: a head read through a dequantization produces a
  different candidate set with no visible symptom, and that case still refuses
  by name.

  **What was wrong was not D12's argument but the PREDICATE the safetensors arm
  used for it.** `SharedHeadSource` read the shared head with one
  `LoadNamedBf16("lm_head.weight")`, so it refused on the STORED DTYPE — and a
  stored dtype cannot separate a head WIDENED into something the target does not
  compute with from a head kept PACKED and computed with natively. Measured on
  `dgx:gpu0` 2026-08-21 under [#1574](https://github.com/mudler/vllm.cpp/issues/1574):
  this engine loads `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a2` and
  generates correctly (the publisher's `19 x 23 -> 437` canary passes, warm
  decode 11.06 tok/s against vLLM's 9.71 on the same box), and adding the DFlash2
  draft dies at `vllm_engine_load: dflash: target tensor lm_head.weight is not
  BF16 (got U8)`. That refusal blocked the only speculative arm the campaign's
  subject has.

  **The merged oracle does not refuse it at all.** The guard D12 ported was live
  at PR head `66e5414c6d75a8529473d977f7458c140bbab8a0` and is GONE at the MERGED
  head `b389ac29465b33f9e9c534df221ea3c129e9793f`: `compute_candidates`
  (`vllm/model_executor/models/qwen3_dflash2.py:282-287`) is now
  `self.candidate_logits_processor.get_top_k_tokens(self.lm_head, hidden_states,
  top_k)` with no `isinstance` test anywhere in the file, and `get_top_k_tokens`
  (`vllm/model_executor/layers/logits_processor.py:241-286`) reaches `_apply_head`
  (`:132-142`) -> `lm_head.quant_method.apply` — the same call
  `LogitsProcessor.forward` makes for the target's own logits. Upstream's answer
  is therefore "compute the top-K through the head the target computes with", and
  it needs no branch to say so because its head is an `nn.Module`.

  **SGLang is a FALLBACK on the same shape, and it is not corroboration.** The
  decision above rests on vLLM alone, which is the primary mirror wherever it
  implements the behaviour, and this observation changes nothing about it. It is
  recorded because it was measured and because the earlier wording read it the
  wrong way round: `DFLASH draft greedy head kept eager (reason=quantized
  lm_head)` means SGLang DECLINED to graph that head on a quantized checkpoint,
  so it is a second engine stepping AROUND the case rather than a second engine
  reaching the head natively. Provenance, because a competitor line without one
  is not evidence:

  - Run by the OPERATOR on `dgx:gpu0` inside an `rc hold` lease, 2026-08-21, for
    campaign [#1574](https://github.com/mudler/vllm.cpp/issues/1574).
  - Image `lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6`
    plus the DFlash2 overlay built from `r0b0tlab/qwen38-27b-nvfp4-sm121-sglang`'s
    `docker/Dockerfile.dflash2`, tagged `qwen38-27b-sglang-dflash2-sm121:0.2.0`.
  - Checkpoint `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @
    `36f717a22990e82c54c1d48ee77c491b87825680`, four shards sha256-verified
    against the publisher's `final-sota-shards.sha256`; draft
    `z-lab/Qwen3.8-27B-DFlash2`.
  - Launched by that repository's own `scripts/serve.sh dflash2`; the line
    appears in `docker logs` at `[2026-08-21 16:43:32]`.
  - Evidence: the operator's measured-results comment on
    [#1574](https://github.com/mudler/vllm.cpp/issues/1574).

  **What lands.** `LoadDflashSharedLmHead` (`qwen3_dflash.h`) asks the TARGET
  loader's own routing question — `DenseLmHeadTakesNvfp4`, the predicate
  `LoadDenseLmHead` routes on, exported so there is one predicate and not two
  descriptions of one — and a ModelOpt/compressed-tensors NVFP4 head lands PACKED
  in `Qwen3DFlashWeights::lm_head_fp4`. `DflashLogitsF32D` routes it through
  `dense_nvfp4::MatmulNvfp4W4A16D`, the SHARED W4A16 dispatcher. It is not the
  same function the target's own head takes -- the target takes
  `MatmulNvfp4F32D` (`qwen3_5.cpp:3106`) -- and `## Owed` O29 records the four
  differences between the two Marlin bodies and which single one of them a head
  can reach.
  `RefuseQuantizedDflash2LmHead` is unchanged in code: it reads
  `lm_head_dequantized`, which only `LoadGgufSharedEmbedAndHeadBf16` sets, so the
  container that still widens a head still refuses.

  **The gate is the PROPERTY and not the load.** "It loaded" would pass while the
  candidate set was subtly wrong, which is exactly the silent failure D12 names.
  `tests/vllm/models/test_qwen3_dflash2_draft.cpp` measures the draft's block
  forward against `Qwen3_5MTPModel::ComputeLogits` — the OTHER draft that shares
  the target's head, whose body for a packed head is the one line
  `DenseLogitsF32D` runs — and requires the logits BITWISE equal and therefore
  the selector's top-K ids and values identical. Non-vacuity is measured in the
  same case: a second packed head with different codes and group scales moves the
  candidate ids, and so does the draft checkpoint's own bf16 `lm_head`. The
  fixture is built from power-of-two group scales and an exact E2M1 code set so
  the dequant is exact and the comparison can be bitwise rather than tolerant.

  **An FP8 head is still refused**, because this engine has no native arm for one
  — `LoadLmHeadAnyDtype` would widen it, which is D12's case exactly. The
  refusal now means "this storage form cannot be computed with", not "this head
  is quantized".

  **AND THE PROPERTY IS REACHED FROM THE LOADER.** The row's fresh review
  measured that every case above entered at `LoadDflashSharedLmHead` or at the
  draft forward, so restoring the pre-row read at
  `src/vllm/entrypoints/model_loader.cpp` — `*head = LoadNamedBf16(*shards_,
  "lm_head.weight", true);`, the exact line #1628 reports as the defect —
  compiled clean and left all three DFlash2 suites GREEN. That is `AGENTS.md`
  `## Nothing lands dead`: a gate nobody can turn red by putting the bug back
  measures a function rather than a capability. Three cases now drive
  `LoadedEngine::FromModelDir` against an on-disk Qwen3.5 dense target whose
  `lm_head.weight` is ModelOpt NVFP4 and an on-disk DFlash2 draft named by
  `--speculative-config`, which is #1628's own reproduction, and they LOAD and
  then GENERATE: the packed arm, the BF16 control arm, and the `VT_LMHEAD_FP4=0`
  rollback, which must refuse by name and must not draft. Restoring that
  `LoadNamedBf16` line takes the suite to 35/36 with four failed assertions, and
  deleting the packed `draft_vocab_size` branch — the second call site the review
  found ungated, which only a PROPOSE reads — crashes the packed case rather than
  passing it. Both were measured on this branch and restored byte-for-byte.

  The four packed-arm cases now name the arm they measure with a scoped
  `VT_LMHEAD_FP4` rather than inheriting the ambient value. Under the documented
  rollback they used to throw `not BF16 (got U8)` and read as four broken tests
  while the behaviour was correct and failing closed; the rollback has its own
  case instead.

## Owed

Everything here is landed-but-not-yet-reachable, or found and not fixed. Each
entry names what is missing, why it is not fixed where it was found, and the
wave that discharges it. AGENTS.md `## Nothing lands dead` permits a staged
slice to land unreached only when this list, the commit body and the pull
request body all name it, so this section is the record that permission depends
on and not a summary.

**W2 DISCHARGED O1, O2, O3 and O4.** They are kept below, struck through in
prose rather than deleted, because the reason each existed is what a later reader
needs and `.agents/completed/` is for superseded documents rather than for four
list items.

- **O1 — DISCHARGED by W2.** The `is_causal` half of W1 was INERT at its own
  merge commit, on both container arms: every artifact that declared the key also
  declared the DFlash2 markers W1 refused, so no checkpoint W1 admitted could
  take any of the three arms. W2 admits `z-lab/Qwen3.8-27B-DFlash2` — which
  declares `is_causal false` beside five `sliding_attention` layers — through
  `CheckDflash2DraftArm`, and `MakeQwen3DFlashDraftConfig` can now parse it (O3),
  so the rule is REACHED by a published checkpoint. Gated in
  `tests/vllm/models/test_qwen3_dflash2_draft.cpp`, which drives the whole
  published `config.json` through the production builder and asserts all five
  layers resolve non-causal. The GGUF arm's inertness is unchanged and moves with
  W5.
- **O2 — DISCHARGED IN PART by W2, and the remainder is O5.** W1's fresh reviewer
  proved the loader's own call sites were not gated for the causality carry.
  `LoadQwen3DFlash` is now driven from a test over a REAL on-disk safetensors
  shard with the published tensor names, which is the function `LoadDflashDraft`
  calls, so the weight half is gated. What is still not gated is the part of
  `LoadDflashDraft` that lives inside the loader's anonymous namespace — see O5.
- **O3 — DISCHARGED by W2.** `MakeQwen3DFlashDraftConfig` could not parse either
  published DFlash2 `config.json`: it did `c.at("rope_theta")` and
  `c.at("block_size")` while both drafts nest them as `rope_parameters.rope_theta`
  and `dflash_config.block_size`. Both are now fallbacks rather than
  replacements — the flat spelling is read FIRST, so every DFlash1 draft is
  unchanged — and the RoPE default is upstream's own
  `set_default_rope_theta(config, default_theta=1000000)`. Red-first: the case
  driving the published Muse-Glimmer document threw
  `[json.exception.out_of_range.403] key 'rope_theta' not found`, quoted in the
  W2 commit body.
- **O4 — DISCHARGED by W2, as ONE change rather than three.** (a) `layer_types`
  is optional, mirroring `getattr(config, "layer_types", None)`, so
  `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` no longer throws
  `key 'layer_types' not found` and #1366's `use_swa` rule resolves for it —
  five layers, every one `causal=0` with `sliding_window=1024`, which is
  upstream's docstring row. (b) `dflash_config.attention_sink_bias` (and the
  top-level `add_swa_attention_sink_bias` upstream falls back to) is REFUSED BY
  NAME, because this lane has no attention sink and landing (a) alone would have
  converted a loud parse error into a quiet wrong answer. A FALSY value is
  upstream's default and is not refused. (c) O4's third fact stands unchanged and
  is not repaired by this wave: MiMo's target `MiMoV2ForCausalLM` is still
  `INVENTORIED` and unimplemented, so no production entry point can serve the
  model that draft heads. The parse is correct and the refusal is loud; the
  drafter is still not runnable, and that belongs to the MiMo model row.

- **O5 — PART DISCHARGED by W4, STRUCTURALLY rather than by a gate, and one
  item remains.** Owner: this row.

  **Item 1, `conv_block_size = draft->k + 1`: the SILENT-WRONG is gone; the line
  is still not driven by any gate, and W4 measured that rather than claiming
  otherwise.** What made this line dangerous was never the line — it was that
  `LoadQwen3DFlash` ALSO wrote the field, from the checkpoint's own
  `dflash_config.block_size`, so deleting the loader's assignment left a
  PLAUSIBLE block behind and the conv masked its taps against the wrong one:
  acceptance-only, token-invisible, and W2 measured every suite staying green
  under exactly that mutation. W4 removes the seed
  (`src/vllm/model_executor/models/qwen3_dflash_weights.cpp`, in the `has_taps`
  branch). The field is now 0 until whoever knows the resolved `k` writes it, and
  `Qwen3DFlashModel`'s existing `VT_CHECK(weights.conv_block_size > 0)` turns a
  dropped assignment into a LOUD refusal on the first DFlash2 forward. That is
  the same remedy W3 applied to the `lm_head_dequantized` carry: remove the
  mutation's SHAPE rather than add a test that first has to reach an
  anonymous-namespace function. The removal itself is gated —
  `tests/vllm/models/test_qwen3_dflash2_draft.cpp` drives a real on-disk DFlash2
  shard through the production loader, asserts `conv_block_size == 0` beside a
  `block_size` the checkpoint DOES declare, and asserts the forward refuses by
  name without it and runs with it; restoring the seed reds that case (1 case /
  2 assertions). **Deleting the loader's line is still GREEN**, measured on
  2026-08-20 across `test_qwen3_dflash2_draft` 27/27, `test_dflash2_runner_reach`
  3/3 and `test_dflash2_draft_routing` 12/12, because the only harness that
  reaches a real engine enters through the in-memory `DflashDraft` overload,
  which bypasses `LoadDflashDraft` by construction. A gate on the line itself
  still needs a draft read off DISK through that function against a live target;
  what changed is that its failure mode is now a refusal rather than a wrong
  answer.

  **Item 2, the conv-geometry NOTICE beside it, is unchanged and still owed.**
  Its cost is a missing diagnostic, not a wrong answer, and it is witnessed by no
  gate for the same reason.

  **Item 3, the `lm_head_dequantized` carry, was DISCHARGED by W3** by removing
  the defaulted parameter, so dropping the argument is a compile error.

  The struck record of what this entry said before W4 is kept below, because the
  reason it existed is what a later reader needs.

- **O5 (the W2/W3 text) — `LoadDflashDraft`'s own DFlash2 lines are not gated.** Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314). Mutation-proven by W2
  on 2026-08-19: deleting `draft->weights.conv_block_size = draft->k + 1;` from
  `LoadDflashDraft` (`src/vllm/entrypoints/model_loader.cpp`) COMPILES CLEAN and
  leaves both focused suites GREEN (`test_qwen3_dflash2_draft` 16/16, 108
  assertions; `test_dflash2_draft_routing` 12/12, 33 assertions). What holds the
  conv's block today is the test setting the field itself, so the gate measures
  `LoadQwen3DFlash` and the forward, not the loader that wires them. It is not
  repaired where it was found for the reason W1's O2 gave and W2 confirmed:
  `LoadDflashDraft` is `static` inside the loader's anonymous namespace, and an
  entry-point gate on it has to load a draft off a LIVE target, sharing that
  target's `embed_tokens` and `lm_head`. **W3 UPDATE: the harness this was
  waiting on now exists** — `test_dflash2_runner_reach.cpp` drives a real
  `LoadedEngine` into `propose_drafts_block` — but it enters through the
  in-memory `DflashDraft` overload, which BYPASSES `LoadDflashDraft` by
  construction, so the line is still ungated and the test sets the field itself
  exactly as the model suite does. Closing O5 needs a draft read off DISK through
  `LoadDflashDraft` against a live target, which is a smaller step now than it
  was and is still not this wave's. The consequence if it regressed is bounded
  and named: the conv would mask its taps against the checkpoint's DEFAULT block
  instead of the resolved `k`, which is invisible unless the two differ — and
  acceptance-only, token-invisible, when they do.

  **W3's fresh review corrected the SIZE of this entry: it is not one line.** O5
  said "`LoadDflashDraft`'s own DFlash2 lines" and then named exactly one, and
  the wave's `## Now` inventory repeated the singular. Inside that one
  anonymous-namespace function the DFlash2 surface is three things:

  1. `draft->weights.conv_block_size = draft->k + 1;` — ungated, above. Owned by
     W4.
  2. The **startup notice** printed beside it. Ungated for the same reason, and
     it went WRONG rather than merely unmeasured: through W3 it still read "the
     candidate selector is NOT implemented and this draft will be refused by name
     at its first propose (SPEC-DFLASH2 W3, #1314)" — naming the wave that had
     just shipped the selector as still owing it, and contradicting
     `CheckDflash2DraftArm`'s corrected notice, which a user loading a real
     DFlash2 directory gets in the same breath. Neither gate could see it:
     `test_dflash2_runner_reach.cpp` captures `cerr` around the IN-MEMORY
     `LoadedEngine` overload, which bypasses `LoadDflashDraft` by construction,
     and `test_dflash2_draft_routing.cpp` drives only `ResolveSpecConfig`.
     **REPAIRED, by deletion rather than by rewording.** The boundary has exactly
     one owner — `CheckDflash2DraftArm`, which runs ahead of every path that
     reaches `LoadDflashDraft` (`ResolveSpecConfig` for the constructor, and
     `FromModelDir` before any load) and whose notice IS gated, in two suites —
     so this line now prints only the resolved conv geometry, which is the one
     thing it knows and the notice above it cannot report because it runs before
     any weight is read. A second ungated copy of the boundary text would go
     stale again at W4 and at W5; deleting it is what stops that, and rewording
     it would not have. What stays owed is that the geometry line itself is
     witnessed by no gate. Its cost is a missing diagnostic, not a wrong answer.

     **The deletion NARROWED the discriminator, and W3's second fresh review is
     right that this is worth writing down.** The two notices never shared a
     trigger. `CheckDflash2DraftArm` classifies on the `architectures` list
     containing `DFlash2DraftModel`
     (`include/vllm/config/speculative.h:133`), while the deleted loader line
     fired on `Qwen3DFlashWeights::IsDflash2()`, which is `conv_taps > 0`
     (`include/vllm/model_executor/models/qwen3_dflash.h:216`). So a safetensors
     draft that carries the convolution tensors but does NOT declare the
     architecture gets no startup notice where it previously got one. That is a
     lost diagnostic and not an unwarned user: `RefuseDflash2PathWalk` is keyed
     to `IsDflash2()` too and still refuses such a draft BY NAME at its first
     propose, and upstream selects the speculator by architecture as well, so a
     checkpoint in that shape is out of contract on both sides.
  3. `shared.LoadInto(&…embed_tokens, &…lm_head, &…lm_head_dequantized)` — the
     wiring that gives D12's `RefuseQuantizedDflash2LmHead` its trigger. **Found
     by mutation and not by reading, and REPAIRED STRUCTURALLY.** Deleting the
     third ARGUMENT compiled clean and left all 38 dflash/gguf suites green after
     a full `libvllm` relink, including `test_qwen3_dflash2_draft` 214/214 and
     `test_gguf_keep_quant` 6093/6093, every one `Status: SUCCESS!`. W3 gated
     both ENDS of this carry — the setter reddens, the reader is gated directly —
     and not the LINK, which is the same "found by mutation, not by reading"
     class the wave claims to have swept and the same rule W3 invoked to correct
     W2. The cause was a defaulted parameter: `SharedHeadSource::LoadInto`
     declared `bool* head_was_quantized = nullptr`, so dropping the argument
     silently turned the carry off and `lm_head_dequantized` stayed false. **The
     default is gone.** Both callers name the argument, the DSpark one passing an
     explicit `nullptr` with its reason, and deleting it is now a COMPILE ERROR
     rather than a green run. This removes the mutation's shape rather than
     adding a test, because any test that could catch it would first have to
     reach the ungated function this entry exists to record.
- **O6 — DISCHARGED on 2026-08-20 by the operator's lease. The CUDA arm of
  `vt::DFlashGroupedConv` COMPILES AND RUNS.** Evidence:
  [#1489](https://github.com/mudler/vllm.cpp/issues/1489), an `rc` job on
  `dgx:gpu0` (GB10, sm_121a, `nvcc` 13.0 matched to the driver) against this
  row's W3 head `b29b6f8869a9eeacc451647e859498491ef6bf1e`. That run reports
  `BUILD_RC=0` and `COMPILE_ERRORS=0` for `test_ops_dflash2_grouped_conv`,
  `test_ops_dflash2_selector_edges`, `test_ops_topk_values_indices`,
  `test_qwen3_dflash2_draft`, `test_dflash2_runner_reach` and
  `test_dflash2_walk_refusal`, and — the precondition that makes a green a result
  rather than a skip wearing a pass — **zero `no CUDA backend; skipping` lines
  across every suite**. The grouped convolution's CUDA==CPU bit-identity case is
  one of the five suites that passed. The per-suite counts live in #1489; this
  entry does not restate numbers it did not take.

  The struck text below is what this entry said before that run, kept because the
  reason it existed is what a later reader needs. ~~The kernel and its
  registration are written and reviewed (`src/vt/cuda/cuda_ops.cu`,
  `DFlashGroupedConvKernel` / `DFlashGroupedConvKernelCuda`), and the CUDA==CPU
  bit-identity case exists and is written to run
  (`tests/vt/test_ops_dflash2_grouped_conv.cpp`, six shapes covering both
  published blocks in bf16 and the modulo arm in f32). It has NEVER COMPILED: the
  authoring host has no `nvcc`, so the CUDA case reports `no CUDA backend;
  skipping CUDA dflash2-grouped-conv parity` and every one of the file's
  assertions runs on CPU. Two specific things are unproven rather than merely
  unrun: that the kernel compiles at all, and that `__fadd_rn`/`__fmul_rn` plus
  `ResRound` reproduce the CPU reference BIT-FOR-BIT on the f32 arm, where the
  intrinsics are the only thing forbidding an FMA contraction the CPU build pins
  off.~~ Both are now measured. The authoring host still has no `nvcc`, so the
  CUDA case still reports `no CUDA backend; skipping` HERE; that is a property of
  this box and no longer of the kernel.

  **What the wave's second fresh review corrected here.** The sentence above used
  to read "the file's 9410 assertions are all CPU". That was true and still read
  as coverage the file did not have: all 9410 were also f32, and on f32 the
  kernel's per-step rounding is the IDENTITY by construction. So the wave's
  central numerics claim — per-step rounding, which is the entire reason the CUDA
  arm is specified BIT-IDENTICAL rather than within an envelope — had no
  executing assertion on either side. The reviewer proved it rather than read it:
  replacing the bf16 branch of the `round` lambda in `src/vt/cpu/cpu_ops.cpp`
  with `return v;` compiled clean and left BOTH focused suites fully green
  (`test_ops_dflash2_grouped_conv` 6/6 cases, 9410/9410 assertions, `SUCCESS!`;
  `test_qwen3_dflash2_draft` 16/16, 108/108, `SUCCESS!`). The draft suite does
  execute the bf16 branch, but every assertion in it is RELATIONAL between two
  runs of the same kernel, so a rounding-policy change moves both arms together
  and cancels.

  The CPU half is now pinned: two CPU-only bf16 cases, one hand-computed against
  literals that differ from the round-once-at-the-end answer in six of eight
  outputs, and one bit-exact at three shapes against a reference that rounds
  where UPSTREAM materializes rather than where our kernel does. Under the same
  mutation they fail: 8 cases / 2 failed, 9930 assertions / 225 failed,
  `Status: FAILURE!`. What is still owed is unchanged in kind and smaller in
  size: CPU == CUDA bit-identity remains unpinned on BOTH sides, because the CUDA
  arm has still never compiled or run here.
- **O7 — DISCHARGED by W3, and its stated REASON was wrong.** W2 recorded that no
  production call site of the DFlash2 refusal was gated — correctly: deleting the
  call in `GPUModelRunner::propose_drafts_block` left every suite green. What it
  also recorded, and what was NOT correct, is why: "a gate therefore needs an
  on-disk TARGET plus an on-disk draft driven through the loader, a step that
  captures the target's aux multi-tap, and a populated per-request device KV
  store. That is the harness O5 is already waiting on, and the one W4 builds."

  It needed none of that. What was missing was a way to hand a DFlash draft to a
  `LoadedEngine` built from IN-MEMORY weights — the exact seam `mtp_weights`
  already had, whose own header comment gives the identical argument ("a
  synthetic spec engine could only ever run with a NULL drafter, which is exactly
  the state a depth gate must not mistake for working speculation"). W3 adds that
  overload, and everything downstream is the production path unchanged:
  `ResolveSpecConfig`, `CheckDflash2DraftArm`, the aux-multi-tap refusal,
  `set_dflash_draft`, `propose_drafts` -> `propose_drafts_dflash` ->
  `propose_drafts_block`.

  `tests/vllm/v1/spec_decode/test_dflash2_runner_reach.cpp` drives a real engine
  over a synthetic Qwen3.5-dense target and an in-memory DFlash2 draft, generates,
  and asserts the WALK REFUSAL's text — which carries the lattice the selector
  produced (`scored-transitions=27 requests=1 steps=3 top_k=3`). That one string
  proves three things no exit status could: the runner entered
  `propose_drafts_block`, the block forward ran there, and the CANDIDATE SELECTOR
  ran there, because those counts ARE the selector's output. Delete the
  `Dflash2SelectCandidates` call in `runner.cpp` and pass a default-constructed
  state — which compiles — and all four read zero.

  The recorded reason is corrected rather than annotated, for the reason W2 gave
  when it corrected its own: AGENTS.md `## Nothing lands dead` grants the
  staged-slice exception only when this list is accurate, so a wrong reason here
  is a defect in the permission and not a wording problem. It also cost W1 and W2
  a wave each of unreachable production code that a fifteen-line overload would
  have gated.

  W3 ALSO removes the duplicate that made the shape possible. W2 had two copies
  of the post-forward step and only the test-reachable one was gated; both propose
  paths now call ONE `Dflash2SelectCandidates`, so the code a user arrives through
  and the code a gate drives are the same implementation.

- **O8 — ANSWERED by W3: the shortcut MIRRORS upstream, and there is nothing to
  repair.** Raised by W2's second fresh reviewer, owned by W3/W4, closed here on
  evidence rather than deferred.

  The premise was that "upstream has no analogue of this precompute: its context
  K/V is whatever the earlier block forwards wrote, and under DFlash2 those
  forwards wrote from a CONV'd stream." That premise is FALSE, and it was read
  from our side of the port rather than from upstream's. Upstream has exactly this
  precompute: `DFlashQwen3Model.precompute_and_store_context_kv`
  (`vllm/model_executor/models/qwen3_dflash.py:590-660` @ vllm-project/vllm#52816
  head `66e5414c6d75a8529473d977f7458c140bbab8a0`) projects the context K/V
  through `_project_context_kv` (`:547-576`), which runs ONE
  `ops.rms_norm(context_states, self._hidden_norm_weight)` and then ONE fused
  `F.linear` against `self._fused_kv_weight` — every layer's K/V from one shared
  normed tensor, which is precisely what
  `Qwen3DFlashModel::PrecomputeContextKVDevice`
  (`src/vllm/model_executor/models/qwen3_dflash.cpp`) does. It applies no
  convolution at any layer either.

  And `DFlash2Qwen3Model` does not override it. The DFlash2 file overrides
  `decoder_layer_cls`, `__init__` and `embed_input_ids`, and adds the selector;
  it contains no `precompute`, no `_project_context_kv`, no
  `_build_fused_kv_buffers`. `DFlash2Speculator` extends `DFlashSpeculator` and
  inherits the propose that calls it. So under DFlash2 upstream's context K/V is
  ALSO projected from an unconvolved shared tensor, and our shortcut is not a
  shortcut at all — it is the port.

  What the reviewer was right about is that nobody had checked, and that if it
  had been wrong the symptom would have been acceptance-only and token-invisible.
  The check is now recorded here so the next reader does not repeat it. Nothing
  in the context path changes in W3.

- **O9 — `dflash_config.input_embedding_scale` is REFUSED, not implemented.**
  Owner: this row, discharged by a later wave or by the first checkpoint that
  needs it. Issue [#1314](https://github.com/mudler/vllm.cpp/issues/1314).
  Upstream scales the draft's token embedding by it in
  `DFlash2Qwen3Model.embed_input_ids` (@ the PR head). NEITHER published DFlash2
  draft declares the key, so implementing it would land a third call site in each
  of this engine's three layer bodies with no checkpoint able to reach any of
  them — AGENTS.md `## Nothing lands dead` — and IGNORING it would run a quietly
  different model on the first checkpoint that sets it, acceptance-only and
  token-invisible. So `LoadQwen3DFlash` refuses a DECLARED value that is not
  upstream's default, by name, with the polarity W2 set for
  `dflash_config.attention_sink_bias` (a falsy/default value is upstream's own
  no-op and is not refused). Gated in
  `tests/vllm/models/test_qwen3_dflash2_draft.cpp`. The cost is bounded and
  named: a checkpoint that sets it does not load, rather than loading wrong.

  **W4 RE-ASKED and the answer is unchanged.** W4 was asked to implement it if
  the wave made it reachable. It does not: the walk consumes the selector's
  lattice, it touches no embedding, and neither published DFlash2 draft has
  gained the key. Implementing it would still land a third call site in each of
  the three layer bodies with no checkpoint able to reach any of them, so the
  refusal stands and the entry keeps its owner.
- **O10 — MOSTLY DISCHARGED on 2026-08-20, and what remains is ONE MEASURED
  DIVERGENCE with its own issue.** Owner: this row for the record;
  [#1489](https://github.com/mudler/vllm.cpp/issues/1489) owns the kernel change.
  Row issue [#1314](https://github.com/mudler/vllm.cpp/issues/1314).

  The operator's `rc` job on `dgx:gpu0` (GB10, sm_121a, `nvcc` 13.0) ran both
  arms at W3 head `b29b6f8869a9eeacc451647e859498491ef6bf1e`. **The kernels
  compile** (`BUILD_RC=0`, `COMPILE_ERRORS=0`), **the device arms genuinely ran**
  (zero `no CUDA backend; skipping` lines; `test_ops_topk_values_indices` reports
  **562 assertions on device against 202 on the CPU-only build**, so the parity
  table added 360 and this is not a green skip), and the lattice suite is among
  the five of six that passed.

  **The tie divergence this entry called "the real risk" did NOT materialise.**
  The CPU arm sorts under an explicit comparator while the CUDA arm
  threshold-searches, compacts and fills from the lowest-indexed equals — two
  different algorithms — and they agree. No tie-row assertion failed: not the
  group straddling the k-th boundary, not the group larger than k, not the ties
  inside the kept set, not the `-inf`-saturated row.

  **What DID fail is the NaN row, and this entry has to report it as a failing
  gate rather than as recorded debt.** `test_ops_topk_values_indices`:
  `7 cases | 5 passed | 2 failed`, `assertions: 562 | 550 passed | 12 failed`,
  `Status: FAILURE!`. All twelve failures name the single literal row
  `"NaN sorts first, as torch.topk does"`. The decisive pair is the direct
  cross-arm comparison — `CHECK( gpu.indices[i] == cpu.indices[i] )` reading
  `2 == 1`, and `CHECK( std::isnan(gpu.values[i]) )` reading `false`, at
  `tests/vt/test_ops_topk_values_indices.cpp:421` and `:424` AS THAT FILE STOOD
  AT `b29b6f886` (this wave's repair moves both lines, so the numbers are quoted
  against the commit that produced them and not against the tree) — so it is a
  genuine backend disagreement and not a wrong expectation on one side.

  The mechanism was derived from the source before those results were read, and
  they agree with it: `TopKValuesIndicesRowKernel`
  (`src/vt/cuda/cuda_sample.cu:548-698`, the kernel body) brackets with `fmaxf`/`fminf`, which
  return the non-NaN operand, and selects survivors with `r[j] > thr`, which is
  false for a NaN. The kernel therefore CANNOT select a NaN, whatever the
  threshold converges to.

  **REPAIRED BY NARROWING, not by a spec paragraph.** AGENTS.md `## Gates` says a
  rule has exactly one result and that a permanent report-only state is not one;
  a suite that reds on every CUDA build is a failing gate, and recording the
  divergence while shipping the red assertion was reporting two results at once.
  So the NaN row is now excluded BY NAME from both device cases
  (`kNanRowName` / `RunsOnCuda`, `tests/vt/test_ops_topk_values_indices.cpp`) and
  kept in `LiteralRows()`, because the CPU order IS the guarantee and is
  mutation-proven. `include/vt/ops.h` states the asymmetry where it states the
  contract, instead of asserting an ordering no shipped backend delivers. The
  exclusion's own match count is asserted on the CPU arm — `CpuOnlyRowCount() ==
  1` — because a filter matching zero rows or every row is invisible on a host
  with no device, and WHICH row it selects is asserted against the row's own
  `want_val` holding a NaN rather than against the name string, so the check is
  not the name compared with itself. Both shapes are mutation-proven: forcing
  `RunsOnCuda` to `true` gives `7 cases | 6 passed | 1 failed`,
  `assertions: 210 | 208 passed | 2 failed`, `Status: FAILURE!`, and forcing it
  to `false` gives `210 | 203 passed | 7 failed`, `Status: FAILURE!`; restored,
  `7/7` and `210/210`, `Status: SUCCESS!`.

  **What is OWED, and to whom.** #1489 owns reconciling
  `TopKValuesIndicesRowKernel` to the NaN-first contract, which is what makes
  `include/vt/ops.h` true on both arms and lets the row go back into the device
  cases. It needs a lease to verify and is not attempted from a host without
  `nvcc`. The cost of leaving it is bounded and named: no shipped path feeds this
  op a NaN logit — the candidate values come from a target LM head — so the gap
  is in the contract's reach and not in any output a user can obtain.

  ~~Same shape as O6 and the same host: both kernels and both registrations are
  written, and both parity cases exist and are written to run
  (`tests/vt/test_ops_dflash2_selector_edges.cpp`,
  `tests/vt/test_ops_topk_values_indices.cpp`), but the authoring host has no
  `nvcc`, so neither has ever compiled and both report `no CUDA backend;
  skipping`. Two things are unproven per kernel rather than merely unrun. For the
  lattice: that it compiles at all, and that the warp-shuffle contraction lands
  inside the 1e-4 relative envelope the case asserts at rank 256. For the top-k:
  that it compiles at all, and — the real risk — that its TIE handling agrees
  with the CPU reference.~~ All of that is now measured. The authoring host still
  has no `nvcc` and still prints `no CUDA backend; skipping` for both cases; that
  is a property of this box.

  **W3's fresh review proved this entry's own MITIGATION false, and the
  mitigation is now real.** O10 used to end "the hand-written tie cases in the
  CPU file are written to exercise exactly that path when the CUDA arm can
  finally run them." They were not, on two counts:

  - Every hand-written tie case called `Run()`, which builds `Queue{Cpu(),
    nullptr}`. Not one of them could ever reach a device, whatever host it ran on.
  - The single case that DID touch the device ran four LCG shapes, under a
    comment claiming "the LCG repeats values at this width". It does not.
    Reproducing that generator bit-for-bit in exact float32 — in Python and again
    in C++, which agree — gives **zero duplicate values in any row** of all four
    parameter sets `{4,513,16,0} {2,128,8,0} {3,200,16,24} {1,64,16,0}`
    (513/513, 128/128, 200/200 and 64/64 distinct), with the k-th largest at
    multiplicity **1** in every row. The same is true of the bulk-sort case's
    `{5,257,16}` at seed `0xC0FFEE`. So the device arm was gated on data
    containing no tie, and the divergence this entry calls the real risk would
    have stayed unmeasured on the day `nvcc` arrived.

  The literal rows now live in one `LiteralRows()` table that **both arms
  iterate** (`tests/vt/test_ops_topk_values_indices.cpp`): ties inside the kept
  set, a tie group STRADDLING the k-th boundary, a tie group larger than k, a
  `-inf`-saturated row (where the CUDA arm's `cur` has to drop below the
  threshold because the group is exhausted), the padded row, and NaN. One CUDA
  case runs the table on device against the LITERALS — two implementations
  agreeing on a wrong tie rule is what asserting one against the other cannot
  catch — and the parity case asserts the two arms against each other over the
  same table plus the four bulk shapes, which cover the bracket at widths the
  literals do not reach and say nothing about ties. Both still report
  `no CUDA backend; skipping` on this host, and both RAN on the GB10 under #1489:
  the tie rows agreed and only the NaN row did not, which is why that one row is
  now the only member of the table the device cases skip.

  **Two more top-k divergence modes this entry did not name, both found by
  reading the kernels rather than running them.**

  - **NaN ordering, and it was undefined behaviour on the CPU side.** The CPU
    comparator read `if (src[a] != src[b]) return src[a] > src[b]; return a < b;`.
    Against a NaN the first test is TRUE and both `>` are FALSE, so NaN compared
    EQUIVALENT to every value while those values are not equivalent to each
    other — an intransitive equivalence, which is not a strict weak ordering and
    is UB in `std::partial_sort`, not merely a surprising answer. **FIXED**: NaN
    is now handled explicitly and sorts FIRST, which is
    `torch.topk(largest=True)`'s own order, and a table row pins it (red before
    the fix: the row returned indices `{2, 1, 3}` where the contract is
    `{1, 2, 3}`). No shipped path feeds this op a NaN logit, so the row is
    synthetic in the same sense the padding row is. **The CUDA arm is NOT
    reconciled to it, and the divergence is now MEASURED rather than expected**:
    `fmaxf` / `fminf` return the non-NaN operand and `r[j] > thr` is false for a
    NaN, so the threshold search cannot select one, and #1489's device run failed
    twelve assertions on exactly that row. The prediction was right and the
    posture that followed it was not: this entry recorded the divergence and then
    left the assertion in both device cases, which ships a red suite on every
    CUDA build. The device cases now skip the row by name and #1489 owns the
    kernel change; see the head of this entry.
  - **Silent truncation in the CUDA compaction**, which is a different failure
    from a tie disagreement. `TopKValuesIndicesRowKernel` compacts survivors with
    `const int slot = atomicAdd(&sh_count, 1);` and writes only `if (slot < k)`;
    everything past `k` is DROPPED. The comment above it justifies that with "at
    most k-1 of them", which holds only if the bracket converged to the exact
    k-th largest inside `kThreshMaxIter` (64). If it did not, the extras are
    dropped in nondeterministic atomic order, so the failure is intermittent and
    row-dependent rather than reproducible — the shape hardest to attribute once
    a GPU is finally available. #1489's run did not exhibit it — every bulk shape
    and every non-NaN literal row matched — which is evidence that the bracket
    converged inside `kThreshMaxIter` on those shapes and not that the drop
    cannot happen. Naming it is still what this entry can do.


- **O11 — the CUDA arm of `vt::Dflash2PathWalk` is UNVERIFIED.** Owner: this
  row, discharged by the operator's GPU lease. Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314). Same shape as O6 and
  O10 and the same host: the kernel and its registration are written
  (`src/vt/cuda/cuda_ops.cu`, `Dflash2PathWalkKernel` /
  `Dflash2PathWalkKernelCuda`) and the CUDA==CPU BIT-EXACT case exists and is
  written to run (`tests/vt/test_ops_dflash2_path_walk.cpp`, three shapes — a
  sub-warp `top_k` 3, the published 16, and 40 so the strided load and the
  `__shfl_xor_sync` reduction both run past one warp — with a forced exact tie
  group and a forced all `-inf` row layered onto the random fixture, because
  neither shape occurs by chance and O10 recorded what it cost to assume one
  did). It does not run HERE — the authoring host has no `nvcc`, so the case
  reports `no CUDA backend; skipping CUDA dflash2-path-walk parity` — but see
  the closing paragraph of this entry: it compiles in CI on every pull request
  and it has since RUN on `dgx:gpu0`
  ([#1518](https://github.com/mudler/vllm.cpp/issues/1518)).

  Two things are unproven rather than merely unrun, and they are different from
  O6's and O10's. That the kernel compiles at all. And that the PARALLEL
  reduction reaches the SEQUENTIAL rule: the CPU arm scans ascending and keeps
  what strictly exceeds the running best, the CUDA arm strides across lanes and
  then butterflies. Get that comparator wrong in one direction and a tie
  resolves to the wrong slot, which then selects the wrong PREDECESSOR row and
  moves every remaining token of the block — and it is acceptance-only and
  token-invisible, like everything else on this row. The all `-inf` row is the
  same class: both arms have to answer slot 0 rather than "no index", and they
  reach it by the same `top_k` seed rather than by coincidence.

  **W4's FRESH REVIEW MEASURED THE COMPARATOR WRONG, and the wave repaired it
  rather than narrowing the claim.** The reviewer emulated the kernel's exact
  32-lane rule on the host and found the two arms disagreeing on a NaN-bearing
  row: `[NaN,-inf]` read cpu 0 / cuda 1, `[NaN,-inf,-inf]` cpu 0 / cuda 1, and
  `[NaN,NaN,-inf]` cpu 0 / cuda 2, while every NaN-free case agreed — the
  all `-inf` row, the all-NaN row and the forced tie group included. Four written
  claims said the arms were bit-exact (`include/vt/ops.h`,
  `src/vt/cpu/cpu_ops.cpp`, `src/vt/cuda/cuda_ops.cu`, `.agents/kernel-matrix.md`
  row `KERNEL-DFLASH2-PATH-WALK`), and on that row they were not.

  The mechanism is one disjunct. The per-lane scan read
  `if (v > best || (v == best && (int)j < slot))` against the seed
  `best = -inf, slot = top_k`. Because `j` only ascends inside a lane, the
  equality arm is unreachable once the lane has claimed anything; its ONLY
  reachable effect was at the seed, where a lane holding `-inf` compared equal to
  the `-inf` seed and claimed a slot the CPU arm's strict scan refuses. On a
  NaN-free row that is invisible, because a real `-inf` lane and a never-claiming
  lane both collapse to slot 0 in the end. So the disjunct is DELETED: the lane
  scan is now strict `>` only, which is the CPU arm's own rule reached the CPU
  arm's own way, and the lower-slot tie preference stays in the cross-lane
  BUTTERFLY, where it is genuinely needed because lanes combine out of slot
  order. The invariant that makes the butterfly safe is now stated beside it:
  with a strict lane scan, `best == -inf` implies `slot == top_k` on every lane,
  so the tie arm never compares a real slot against a seed.

  **Narrowing the four claims to "bit-exact on any NaN-free lattice" was the
  alternative and was rejected.** It would have been true, and no shipped path
  can feed this op a NaN (the lattice comes from `vt::Dflash2SelectorEdges` over
  a target LM head), so the reachable behaviour is identical either way. It was
  rejected because the divergence is a two-token deletion of a provably dead
  disjunct: recording an accommodation for a defect that costs one line to remove
  is worse than removing it. That is the difference between this entry and O10,
  where `TopKValuesIndicesRowKernel` cannot be made NaN-first without a redesign
  of its pivot bracket, and narrowing was therefore the honest result.

  **What this adds to the debt, precisely.** The CPU half of the strictness claim
  is now gated and mutation-proven: turning `>` into `>=` in
  `Dflash2PathWalkKernel` reddens **3 cases / 5 assertions**, `Status:
  FAILURE!` — `a tie resolves to the LOWEST slot` (2 assertions, got 13 for 11
  and 22 for 20), `an all -inf row resolves to slot 0` (1, got 33 for 31) and
  `dflash2-path-walk: a NaN never wins a slot` (2, got 83 for 81 and 83 != 83),
  in `tests/vt/test_ops_dflash2_path_walk.cpp`. An earlier revision of this
  entry called the NaN case the one a `>=` reduction fails "while still
  answering the tie rows and the -inf row", which contradicted the count printed
  in the same sentence;
  [#1518](https://github.com/mudler/vllm.cpp/issues/1518) corrects it. All three
  fail, because this scan ascends and `>=` keeps the LAST maximum, so the tie
  row answers slot K-1 and the all -inf row claims K-1 instead of leaving the
  seed for the collapse. What the NaN case adds that the other two do not is the
  NaN CLASS, and that it is the row on which the two backends actually
  diverged.

  The CUDA half is not gated on THIS host, which has no `nvcc`, so the case
  still reports `no CUDA backend; skipping` here. It is gated elsewhere, and
  [#1518](https://github.com/mudler/vllm.cpp/issues/1518) corrects an earlier
  revision of this entry that called the deleted disjunct never compiled and the
  repair "a source change made on an argument, not on a measurement". Both
  halves are covered. COMPILE: `src/vt/cuda/cuda_ops.cu` is in the CUDA source
  list in `CMakeLists.txt` (`target_sources(vllm PRIVATE ...)` under
  `if(VLLM_CPP_CUDA)`), which CI's `build-cuda-fat` job builds for ten
  architectures (`80;86;87;89;90a;100a;103a;110;120a;121a`) on every non-closed
  `pull_request` event, so the deletion goes through `nvcc` pre-merge. RUN: the
  operator executed this suite on `dgx:gpu0` (GB10, sm_121a) at the W4 merge
  commit and it reported **83 assertions on device against 49 on CPU**,
  `Status: SUCCESS!`, with zero `no CUDA backend; skipping` lines and
  `CUDA_OBJECTS_BUILT=34`. The increment includes the CUDA parity case — whose
  fixture CHAINS a third forced row, a NaN row at step 2 predecessor 0, reached
  because the tie group and the -inf row each answer slot 0 — which is the row
  that measured the divergence. What this entry still owes is the rest of it,
  not the comparator.

- **O13 — a GGUF drafter is DEQUANTIZED to bf16 at load, so the k-quant arms
  ship without their memory saving.** Owner: this row, recorded rather than
  fixed. Issue [#1314](https://github.com/mudler/vllm.cpp/issues/1314).

  This is the DFlash GGUF lane's own design, established by `SPEC-DFLASH-GGUF`
  and argued in the file comment on
  `src/vllm/model_executor/models/qwen3_dflash_gguf.cpp`: the resolver
  dequantizes wholesale so that the ENTIRE safetensors weight body is reused
  unchanged, which is affordable because a DFlash draft is a handful of layers.
  W5 does not change it, and deliberately: doing so would be a keep-quant
  residency port of the whole DFlash lane, not a DFlash2 wave.

  **W6 MEASURED THE BILL ON A GATE HOST, and O13's claim survives contact with
  a running process.** Both arms were driven through `examples/vllm-cli` against
  the real 27B target on `dgx:gpu0`, same prompt, same k, `/usr/bin/time -v`
  around each:

  | arm | file on disk | peak RSS |
  |---|---:|---:|
  | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1 143 006 752 B (1.06 GiB) | 47 028 616 KB (**44.85 GiB**) |
  | `z-lab/Qwen3.8-27B-DFlash2` safetensors | 3 848 817 896 B (3.58 GiB) | 46 701 608 KB (**44.54 GiB**) |

  The two files differ by 2.52 GiB on disk and the two processes differ by
  **0.70%** — and the QUANTIZED one is the LARGER of the two, by 319 MiB, which
  is the transient cost of dequantizing at load. So "choosing Q4_K_M saves
  download and disk and saves NOTHING at runtime" is now a measurement rather
  than an inference from a tensor table, and the direction of the residual is
  the opposite of the one a reader would assume.

  A SECOND thing that run settles, and it is worth more than the memory number:
  **the two arms propose DIFFERENT drafts.** Block 1 of the same prompt reads
  `[248069 271 760 6511 314 9338 369]` from the GGUF drafter and
  `[248069 271 760 6511 314 11751 25]` from the safetensors one, while both
  emit the same target tokens. The quantization moves the drafter's proposals
  without moving the answer, which is exactly the acceptance-only, token-invisible
  regime this row exists to gate — and it also proves the GGUF arm is not
  quietly reading the safetensors weights.

  What W5 changes is the SIZE of the bill, so it is measured here rather than
  left to a gate host. Summed over `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`'s own
  tensor table on 2026-08-20: 1 924 404 480 elements, so 3 848 808 960 bytes
  (3.584 GiB) resident bf16 against 1 143 006 752 bytes on disk -- a 3.37x
  expansion, where the DFlash1 drafter's ratio at the same quant is smaller
  because it has no selector. 254 279 680 of those bytes are the two codebooks,
  which is `## Risks/decisions` D5's number arriving on the container most users
  run. The consequence is bounded and named: choosing Q4_K_M over BF16 saves
  download and disk and saves NOTHING at runtime. `docs/SPECULATIVE-DECODING.md`
  says so where a user picking a file will read it.

- **O14 — `LoadDflashDraft`'s GGUF branch is still not gated from a production
  entry point.** Owner: this row. Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314). This is O5's shape,
  unchanged by W5 and named again because W5 adds lines to that branch's
  neighbourhood rather than to the branch. `test_qwen3_dflash2_gguf.cpp` drives
  the exact pair `LoadDflashDraft` calls -- `MakeDflashGgufConfig` then
  `LoadQwen3DFlashFromGguf` -- and `test_dflash2_draft_routing.cpp` drives
  `FromModelDir` far enough to prove the classification runs on a real GGUF
  ahead of every path operation. What neither reaches is the function itself,
  for O5's reason: it is `static` inside the loader's anonymous namespace, and
  the only real-engine harness enters through the in-memory `DflashDraft`
  overload. The bounded consequence is unchanged: a dropped
  `conv_block_size = draft->k + 1` leaves 0 and the first DFlash2 forward
  refuses BY NAME, which W4 made true by removing the checkpoint-derived seed.

- **O12 — the PROBABILISTIC draft-sample arm and its realized-q cache are NOT
  ported.** Owner: `SPEC-ACCEPT-VARIANTS` (`.agents/engine-matrix.md`) for the
  configuration and the verify; this row for the DFlash2 half when that lands.
  Issue [#1314](https://github.com/mudler/vllm.cpp/issues/1314).
  `## Risks/decisions` D13 carries the decision and its reason: no configuration
  this engine admits can reach the arm, because `ParseSpeculativeConfigJson`
  refuses `draft_sample_method: "probabilistic"` by name and the verify is
  accept-iff-equal, so landing it would land two mechanisms no production entry
  point can reach.

  **This is a NOT-PORTED entry, not a landed-unreached one**, and the difference
  matters for what the next reader has to do. Nothing of it is in the tree: the
  walk op has no noised arm and no realized-score output, `Dflash2WalkPath`
  returns tokens only, and no `draft_logits` buffer is allocated anywhere. What
  IS recorded is the layout, so it is not re-derived: upstream's
  `DFlash2Speculator.draft_logits_spec` returns `(torch.float32, -inf)`, the
  cache is `[max_num_reqs, num_steps, vocab]`, `_cache_draft_logits_kernel`
  writes ONLY the K candidate columns of `[req_state, step]` after clearing the
  previous step's K, and the values it writes are the RAW edge scores of the
  chosen predecessor row — not temperature-divided and not noised, which is what
  lets the rejection sampler divide by the same temperature and reproduce the
  walk's number bitwise.

  The cost of leaving it is bounded and named: a user who asks for probabilistic
  drafting is refused at config parse with the owning row named, on DFlash2
  exactly as on every other method here. Nothing degrades silently.

  Bit-parity for the noise is its own port and should be scoped as one: the draw
  is Triton's `tl.randint(seed, pos)` seeding `tl.rand`/`tl.randint4x` (Philox
  4x32-10) keyed by the CANDIDATE TOKEN IDS, and `-log(-log1p(-u))` rather than
  the naive `-log(-log(u))` — upstream's own comment says why, and it is a
  precision argument about where fp32 resolves the winning tail, not a stylistic
  one.


- **O15 — the three OUTPUT-SCALAR key spellings on the GGUF container are
  INFERRED, not measured.** Owner: this row for the record, W6 for the
  measurement. Issue [#1314](https://github.com/mudler/vllm.cpp/issues/1314).

  `MakeDflashGgufConfig` reads `dflash.output_multiplier`,
  `dflash.final_logit_softcapping` and `dflash.input_embedding_scale`. The five
  GEOMETRY keys beside them ARE measured — the published file writes the HF key
  name verbatim under the `dflash.` prefix, five of five — and these three
  extrapolate that convention. Re-read on 2026-08-21 across all three published
  arms of `z-lab/Qwen3.8-27B-DFlash2-GGUF`: 47 KV entries each, and NONE of the
  three appears in any of them. So the convention is unconfirmed exactly where it
  is being relied on.

  The cost if llama.cpp's converter spells them otherwise is the class D9 exists
  to prevent: the reader finds no key, `LoadQwen3DFlash` supplies upstream's
  default, and the drafter runs with a multiplier of 1.0 and no softcapping
  against a checkpoint that asked for 0.196 and 20.0. Both are applied to the
  candidate VALUES before the selector scores them, so the top-K reorders and
  ACCEPTANCE moves with the tokens still verifying losslessly — invisible to
  every token gate. It is not fixed here because the only fix that closes it is a
  measurement on an artifact that does not exist yet: `z-lab/Muse-Glimmer-30B-DFlash2`
  is the checkpoint that SETS two of the scalars (#1327) and it has no published
  DFlash2 GGUF conversion. Refusing an unknown `dflash.*` key instead was
  considered and rejected: the published files carry `dflash.context_length`,
  which this reader does not read, so a refusal on unknown keys would refuse
  every artifact that exists. Re-checked 2026-08-21 against the reader itself:
  `dflash.target_layers`, named here through W5 as a second unread key, is in
  fact read and REQUIRED -- `qwen3_dflash_gguf.cpp` `VT_CHECK`s it non-null and
  then undoes llama.cpp's `+1` offset. `dflash.context_length` genuinely is
  unread: `MakeDflashGgufConfig` reads twenty `dflash.*` keys -- seventeen
  spelled literally plus the three optional output scalars above -- and that is
  not one of them, and the only `context_length` reads anywhere in `src/` sit
  under the `laguna.`, `<arch>.` (qwen35moe/qwen3next/qwen35), `deepseek4.` and
  `muse-glimmer.` prefixes. Whether it is the ONLY unread key of the published
  47 is not asserted here, because this checkout has never enumerated them; one
  unread key refuses every artifact exactly as two would, so the disposition is
  unchanged either way. W6 discharges this by converting that draft with
  llama.cpp and reading the keys it writes.

- **O16 — SETTLED by W6 on 2026-08-21, and the answer is neither `==` nor `>=`:
  upstream has NO comparison at all.** Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314). Read directly out of
  the beyond-pin oracle wheel (`vllm-0.1.dev1+g66e5414c6`, sha256
  `fbc247ab1bda93a81ff7a68658cdda65b697e263ad2c43a2bc62c2591d207439`), which is
  the artifact this row's gate runs, so the reading is of the code that executes
  rather than of a page.

  `DFlash2Qwen3ForCausalLM.compute_candidates`
  (`vllm/model_executor/models/qwen3_dflash2.py`, the `num_pad` block just before
  `_topk`) never compares the codebook row count against anything. It prevents
  the out-of-range read STRUCTURALLY:

  ```python
  num_pad = self.lm_head.shard_indices.num_org_vocab_padding
  if num_pad > 0:
      logits[..., -num_pad:] = -float("inf")
  values, ids = _topk(logits, selector.top_k)
  ids = ids.to(torch.int64) + self.lm_head.shard_indices.org_vocab_start_index
  ```

  The head vLLM materialises is `pad_vocab_size(org_vocab_size, 64)`
  (`vocab_parallel_embedding.py`, `DEFAULT_VOCAB_PADDING_SIZE = 64`), so it is
  routinely WIDER than the true vocabulary; `num_org_vocab_padding` is that
  excess, and masking it to `-inf` means no padded column can ever survive the
  top-k. Every id that reaches the codebooks therefore lies in the target's
  ORIGINAL vocabulary range, and the codebooks are sized from the DRAFT config's
  `vocab_size`. Upstream's invariant is an equality between those two ORG
  vocabularies, held by checkpoint pairing and enforced by nothing.

  **So the guard's polarity is right and its OPERAND is the thing to watch.**
  Ours compares against the MATERIALISED head width; upstream's reachable span is
  the width MINUS the padding. The two are the same number here and only here:
  this engine is single-device with no tensor parallelism and pads no head, so
  `Dflash2CandidateArgs::num_org_vocab_padding` is structurally 0 on every path
  that reaches the guard, and the materialised width IS the org span. The new
  refusal class this entry warned about is therefore LATENT rather than live, and
  its trigger is named: the first time this engine pads an `lm_head` — a
  tensor-parallel shard, or a checkpoint that ships padded rows — the comparison
  has to move to `vocab - num_org_vocab_padding` in the same edit, or a target
  vLLM would happily draft for is refused here.

  **Measured on the shipping pair rather than argued**: the target's
  `lm_head.weight` is `[248320, 5120]` (read from
  `model-00018-of-00018.safetensors`'s own header) and both codebooks are
  `[248320, 256]`, so `codebook_rows == vocab` holds, and `248320 % 64 == 0` so
  vLLM pads this head by zero columns as well. The two engines agree on this
  checkpoint by measurement, not by coincidence.

  **The guard is KEPT and NOT relaxed to `>=`.** Relaxing would admit a
  genuinely mispaired draft whose codebooks merely happen to be longer, which is
  a wrong answer rather than a crash — the direction this entry already argued
  is worse. Rewriting it to subtract a quantity that is provably 0 today would
  add an arm no production entry point can reach, which AGENTS.md `## Nothing
  lands dead` forbids. What W6 owed was the READING, and the reading is here.

  The struck text below is what this entry said before the oracle could answer
  it. ~~The codebook-span guard compares `==`, and upstream's own condition has
  NOT been read.~~ Owner: W6. Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314).

  `Dflash2SelectCandidates` requires the predecessor codebook's row count to
  EQUAL the target head's width. The out-of-range read it exists to refuse needs
  only `>=`, so the guard as written is strictly stronger than its own stated
  reason, and W5's fresh review was right to flag the gap between the two
  (#1314 F7). It is left strict rather than relaxed here because the question is
  not answerable in this checkout: the local oracle sits at the parity pin and
  carries no DFlash2 at all, so nothing here can say whether upstream pads a
  target head relative to the draft's codebooks or forbids it.

  The consequence of guessing wrong in either direction is named so W6 does not
  have to re-derive it. Keeping `==` introduces a NEW REFUSAL CLASS on the
  already-shipping safetensors lane: any target whose `lm_head` is padded — to a
  tensor-parallel multiple, or by an added special token — now fails to draft
  where before it read a codebook row that existed. Relaxing to `>=` would admit
  a genuinely mispaired checkpoint that happens to be narrower, which is the
  wrong answer rather than the crash. W6 settles it by reading
  `DFlash2Speculator` at the PR-head oracle and mirroring what upstream does.

- **O17 — DISCHARGED by W6 on 2026-08-21. A published DFlash2 artifact has now
  been LOADED, at its real geometry, through a PRODUCTION ENTRY POINT, and it
  DRAFTED.** Issue [#1314](https://github.com/mudler/vllm.cpp/issues/1314).

  Not a header read and not a unit harness: `examples/vllm-cli` — an ABI client,
  which is what `.agents/reachability.md` means by a production entry point — was
  pointed at `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (1 143 006 752 B, sha256
  `18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594`, recomputed
  from the staged copy) over the real `Qwen/Qwen3.8-27B` target on `dgx:gpu0`,
  greedy, k=7. It loaded, it proposed **7 speculative blocks**, and it emitted 24
  tokens with `finish_reason=length`. Blocks read off the production
  `[SPECTRACE]` line, first three:

  ```
  [SPECTRACE] req=0 pos=6 k=7 ns=2 acc=1 draft=[ 13 271 22916 6970 279 6511 314 ] emit=[ 13 198 ]
  [SPECTRACE] req=0 pos=8 k=7 ns=1 acc=0 draft=[ 248069 271 760 6511 314 9338 369 ] emit=[ 760 ]
  [SPECTRACE] req=0 pos=9 k=7 ns=3 acc=2 draft=[ 6511 314 9564 369 11751 13 198 ] emit=[ 6511 314 9564 ]
  ```

  That closes the gap this entry named: a 248320 x 256 codebook is 127 MB and the
  synthetic fixture's was 128 KB, so a row stride or an offset defect past a size
  threshold would have passed every earlier case. Every published byte of this
  file is now mapped and decoded, at `H = 5120` with 5 layers.

  **The residency it actually paid: peak RSS 47 028 616 KB (44.85 GiB)**,
  measured by `/usr/bin/time -v` around the whole process, whose presence was
  asserted before the run rather than assumed. Wall 11:17 for 24 tokens off a
  cold CIFS mount, which is a LOAD figure and not a decode figure and is recorded
  as neither a speed result nor an axis.

  **THE DISCHARGE IS NOT REPRODUCIBLE FROM THIS TREE**
  ([#1562](https://github.com/mudler/vllm.cpp/issues/1562)), and that is recorded
  rather than left for a reader to discover. No runner script and no log is
  committed for any of it: not the `examples/vllm-cli` invocation, not the
  `/usr/bin/time -v` capture, not the `[SPECTRACE]` lines quoted above. The
  finding stands — the artifact was loaded and it drafted — but the numbers
  cannot be re-derived by anyone who was not on that lease, and neither can the
  safetensors arm they are compared against. What is owed is committing the
  runner, or saying plainly that it was lost with the lease.

  "Drafts in all three published arms" is an inference from the ENCODINGS, and
  the record should say so. What is measured is: a synthetic fixture at
  `H = 256` that writes each arm's block format itself and drafts from it, plus
  an asset-gated case that opens all three published files and reads their
  HEADERS — the KV block and the tensor table, kilobytes rather than the 7 GB the
  three arms weigh. No case maps a byte of published tensor data, and no case
  runs the real geometry (`H = 5120`, 5 layers, a 248320-row codebook). The
  fixture's own coverage is what carries the decode, which is why W5's repair
  wave counts every packed field it drives (#1314 F1).

  The gap this leaves is a shape or an encoding that only the real file has: a
  248320 x 256 codebook is 127 MB and the synthetic one is 128 KB, and a row
  stride or an offset defect that appears only past a size threshold would pass
  everything here. It is not closed in W5 because a full load of the three arms
  is a 7 GB read and a 3.584 GiB residency (O13), which belongs with W6's leased
  GPU run and not in a CPU suite that must stay CI-cheap. W6 loads at least the
  Q4_K_M arm end to end and records the residency it actually paid.

- **O18 — DISCHARGED 2026-08-21. The Q8_0 fixture's block scale now ALTERNATES,
  and W5's stated reason for deferring it was wrong.** Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314).

  W5's repair wave drove every PACKED INTEGER field across its width and counted
  it — the 6-bit scale in both halves of `get_scale_min_k4`, the 6-bit min in
  both, and the 4-bit quant in both nibble positions. The fp16 SCALARS in front
  of them were not covered the same way: `EncodeQ8_0` fixed `d` at `0x1C00` in
  every block of every tensor, and `EncodeQ4_K` fixes `d` at `0x2400` and `dmin`
  at `0x2000`.

  **The reason W5 gave for leaving it — that varying `d` costs the BIT-EXACTNESS
  this suite's comparison rests on — is refuted by the remedy W5 named in its own
  next sentence.** A second POWER OF TWO keeps every decoded value `q / 2^p` with
  `|q|` inside eight significant bits, so it survives the loader's bf16 store
  exactly and the assertion stays equality rather than a tolerance. The cost the
  refusal was built on is not paid, and recording it a fifth time with the owner
  written as "this row" — a row at W5 and closing — would have left it owned by
  no wave at all.

  So it is FIXED here rather than deferred again. `EncodeQ8_0` alternates `d` by
  BLOCK between `0x1C00` (2^-8) and `0x2000` (2^-7); `Q8Coverage` counts the
  blocks written at each, and `CheckQuantArm` asserts BOTH nonzero on every
  COMPARED tensor. Bit-exactness is proved by measurement rather than by the
  argument above: the suite's L2 leg still compares bf16 bit-for-bit and reads
  9 cases / 4746 assertions / `Status: SUCCESS!` / rc 0, up from 4730.

  It was a live gap, not a theoretical one. `DequantQ8_0` reads `d` from EACH
  block header, and rewriting `const float d = ReadF16(blk)` to `ReadF16(data)`
  — the whole tensor decoded at block 0's scale — left the PRE-repair suite at
  9 cases / 4730 assertions / `SUCCESS!` / rc 0, compile rc 0. Against the
  repaired fixture the same mutation reddens 1 case / 7 assertions / rc 1, and
  removing the alternation from the encoder reddens the same 7, so the counter
  is armed rather than decorative.

  What remains uncovered is narrower and stays that way: `EncodeQ4_K`'s `d` and
  `dmin` are still one value each. Those two multiply an already fully driven
  6-bit scale and min, `ReadF16` is a `memcpy` plus the shared `vt::F16ToF32`
  gated across the exponent range by its own suite, and the Q4_K block header's
  offset and width are pinned by the byte-count precondition and by every value
  after them decoding correctly.

- **O19 — CLOSED 2026-08-21, and recorded rather than owed. The coverage
  counters now read the SAME POPULATION the comparison reads.** Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314).

  Through W5 the Q4_K counters were summed over EVERY Q4_K tensor in the fixture
  while L2 compares only the 11 `Dflash2Names()` slots, of which 7 are quantized.
  Nothing was wrong at the shipped fixture — the 7 compared tensors reach the
  full span on their own — but the precondition and the comparison were reading
  different populations, so a change to a NON-compared tensor could have inflated
  the counters while the compared set stopped driving the field. `BuiltGguf` now
  keys coverage per HF tensor name and `CheckQuantArm` merges over
  `Dflash2Names()` alone, which makes the two populations the same by
  construction. `q[j] & 63` → `& 15` still reddens 1 case / 7 assertions under
  the narrowed population, so the narrowing did not mute the Q4_K leg.

- **O20 — CLOSED 2026-08-21, and recorded rather than owed. The asset case now
  reports PER VARIABLE.** Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314).

  The published-artifact case took its skip on `dflash2 == nullptr && dflash1 ==
  nullptr`, so it announced the skip only when BOTH variables were unset. A run
  with `VLLM_DFLASH2_GGUF_MODEL` alone printed "reading 3 published DFlash2 GGUF
  arm(s)" and said nothing at all about the DFlash1 regression half it had just
  skipped — the #1382 shape one level in, where a loud line invites the reader to
  infer that the whole case ran. Each half now reports its own absence. Measured
  2026-08-21 over all four combinations, with
  `VLLM_DFLASH2_GGUF_MODEL=qwen3.8-27b-dflash2-gguf/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`
  and `VLLM_DFLASH_GGUF_MODEL=muse-glimmer-30b-gguf/dflash-kquant.gguf`: neither
  4746, DFlash1 only 4750, DFlash2 only 4941, both 4945, `Status: SUCCESS!` and
  rc 0 in all four, and the unset half named by its own variable in each of the
  first three.


- **O21 — vllm#52816 MERGED at `3406ec1d`, and the port is NOT reconciled onto
  it.** Owner: `SPEC-DFLASH2`. Issues
  [#1538](https://github.com/mudler/vllm.cpp/issues/1538) (the head move) and
  [#1561](https://github.com/mudler/vllm.cpp/issues/1561) (the reconciliation the
  merge makes due). Measured by W6 on 2026-08-21 from `raw.githubusercontent.com`
  at both heads, while taking the gates; RE-MEASURED and restated on 2026-08-21
  by the W6 repair wave.

  **W6 wrote this entry as "the pull request is still OPEN", and it had already
  merged.** `2026-08-21T05:27:22Z`, merge commit `b389ac29`, head `3406ec1d` --
  46 minutes before W6's work commit. So `3406ec1d` is not "a third unmerged
  head" this row may decline to chase; it is what vLLM's `main` now carries, and
  the port is one merge behind upstream rather than one revision behind a branch.

  `## Gates` G2's rule selects the merge commit `b389ac29` for the gate head.
  W6's capture stays pinned to `66e5414c` because that is the wheel that ran,
  and G2 records that as a dated exception. Re-reading G2 and G3 at `b389ac29`
  is #1561 and is owed.

  The delta is +11/−80, +4/−16 and +2/−5 over the three ported files
  (`qwen3_dflash2.py`, `dflash2/speculator.py`, and the BASE
  `v1/worker/gpu/spec_decode/speculator.py`). **All three RE-MEASURED on
  2026-08-21 by the repair wave** with the same method W6 used --
  `git diff --no-index --numstat` over the raw blobs at each head -- and all
  three hold unchanged. The fresh review reported the base file as +3/−6; that
  reading does not reproduce. At `66e5414c` -> `3406ec1d` the base
  `speculator.py` changes only `draft_logits_spec`'s docstring, two lines added
  against five deleted. The DFlash1 `dflash/speculator.py` is byte-IDENTICAL at
  both heads (sha256 `a8f03bbe...`), which is worth recording because it is the
  file a reader is most likely to reach for by that name. And the
  large one is a RELOCATION: `compute_candidates` loses its whole body to
  `LogitsProcessor.get_top_k_tokens`, with `output_multiplier` and
  `final_logit_softcapping` moved into a `LogitsProcessor(vocab_size, scale=,
  soft_cap=)` built in `__init__`, and `_topk` absorbed there too. The padding
  mask, the id rebase, the TP all-gather and the scale-THEN-softcap order are all
  preserved (`logits_processor.py:241-286` @ `3406ec1d`), so the answer is
  unchanged and O16's reading holds at BOTH heads -- and now at MERGED upstream,
  which is the same tree.

  **One behavioural thing is not preserved and it is the one this row mirrors by
  name:** the explicit `UnquantizedEmbeddingMethod`/`UnquantizedLinearMethod`
  guard `## Risks/decisions` D12 ports as `RefuseQuantizedDflash2LmHead` is
  DELETED at `3406ec1d`. Our guard's own reason is independent of upstream's and
  stands — the GGUF arm dequantizes a q6_K/NVFP4 `output.weight` to bf16 and a
  GGUF target with a safetensors DFlash2 draft is an admitted combination here —
  so nothing changes today. **What was owed was "the decision when #52816
  settles". It has settled, onto vLLM's `main`**, so the decision is due rather
  than parked, and it is [#1561](https://github.com/mudler/vllm.cpp/issues/1561).
  Our guard stands on its own reason; what #1561 owes is writing that down
  against a merged upstream rather than against a branch, because "upstream
  deleted it on an unmerged head" and "upstream deleted it on main" are
  different facts.


- **O22 — the row's declared oracle backend rests on a premise a W6 run
  FALSIFIED, and the reconciliation is not this wave's.** Owner:
  `SPEC-DFLASH2` for this row's own denominator, the operator for the oracle
  record. Issue [#1456](https://github.com/mudler/vllm.cpp/issues/1456), where
  the measurement is posted in full.

  `## Gates` and the staged `FA-CONSTRAINT.txt` name `TRITON_ATTN` because
  #1456 measured vLLM's vendored flash-attention emitting `sm_80`/`sm_75` at
  `CUDA_ARCHS=12.0` and concluded the GB10 oracle has no `FLASH_ATTN`
  denominator. The ARCH measurement is not disputed. The CONCLUSION is: W6's
  first capture accidentally ran on `FLASH_ATTN` — the export it used,
  `VLLM_ATTENTION_BACKEND`, does not exist in this wheel — and vLLM loaded the
  27B, captured CUDA graphs including the DFlash2 speculator's own, and generated
  4 x 64 coherent tokens with speculation live (209 accepted of 350 drafted,
  mean acceptance length 5.00). `sm_80` PTX JITs FORWARD;
  `cudaErrorUnsupportedPtxVersion` is the opposite failure.

  W6 does not change the declared backend on that finding. The developer
  recorded `TRITON_ATTN` for this row, and a wave that quietly substitutes a
  denominator is the failure this protocol exists to stop. What W6 does instead
  is take BOTH arms, name each in its own golden, and leave the choice recorded
  rather than assumed.

  **AND THE FLASH_ATTN GOLDEN'S BACKEND LABEL DOES NOT MEET THE RULE THIS ENTRY
  ITSELF LAYS DOWN** ([#1562](https://github.com/mudler/vllm.cpp/issues/1562)).
  The rule above is: pass the kwarg and read the resolved value back off the
  built engine. That golden's label was not read back. It carries
  `attention_backend_source: "corrected from the run log by w6-relabel.py; the
  capture's original value came from VLLM_ATTENTION_BACKEND, which does not exist
  in this wheel and selected nothing"` -- a POST-HOC relabel of a run whose
  recorded value was known to be meaningless. The run log it was corrected from
  is not committed and neither is `w6-relabel.py`, so the label cannot be
  re-derived from anything in this tree. The evidence for `FLASH_ATTN` is the
  wheel's own `Using FlashAttention version 2` startup line as W6 read it, and
  that reading is now unauditable. The TRITON_ATTN golden's label WAS read back
  off the built engine and is not affected.

  **THE RULE IS NOW A COMMITTED REFUSAL, AND THE LABEL IS STILL WRONG.**
  `tools/bench/dflash2_speed_harness.py::attention_backend_reasons` admits
  exactly one provenance, `read_back_from_engine`, AND requires the record to
  name the dotted walk it was read off, checked against `BACKEND_PROBES`.
  `tools/bench/dflash2_oracle_capture.py::resolve_attention_backend` is the only
  thing that emits that pair -- it walks the BUILT engine and REFUSES when no
  probe resolves, rather than falling back to the request kwarg.
  `tests/tools/test_dflash2_speed_harness.py` drives the golden's own relabel
  sentence through the checker and requires the refusal, so the #1562 defect
  cannot be reintroduced verbatim.

  **BE PRECISE ABOUT WHAT THAT BINDS.** An arm record is a JSON file, so a
  person can type both fields, and an earlier draft of this harness claimed the
  admitted value "cannot be produced" by anything else and was therefore "a real
  bind rather than a label anyone can type". It is not: the fresh review typed
  the literal string into a hand-authored record and got a ratio. What the pair
  binds is the SHAPE OF THE CLAIM -- #1562's actual defect was a label corrected
  afterwards from a run log and recorded honestly as such, and that record is now
  refused by name. Passing requires FABRICATING a read-back rather than
  transcribing a relabel, and the protocol answers fabrication with the fresh
  review, not with a checker.

  What none of it does is repair the committed `FLASH_ATTN` golden: its label was
  never read back, the log it was corrected from was lost with the lease, and
  only a RE-CAPTURE can give it a provenance. That capture is owed under O26.

  **Taking both bought something the row needed anyway: A CONTROL ON THE NEAR-TIE
  ENVELOPE, measured on vLLM AGAINST ITSELF.** Same wheel, same host, same
  workload, same k, greedy, `max_num_seqs` 1, FULL decode graphs on both — only
  the attention backend differs. The two arms:

  | | FLASH_ATTN | TRITON_ATTN |
  |---|---:|---:|
  | `spec_decode_num_drafts` | 50 | 47 |
  | `spec_decode_num_draft_tokens` | 350 | 329 |
  | `spec_decode_num_accepted_tokens` | 209 | 216 |
  | accepted / drafted | 0.597 | 0.657 |
  | generated tokens | 256 | 256 |

  The two arms produce the SAME 64 tokens on 3 of the 4 prompts. The one that
  moves is `def fibonacci(n):`, and it diverges at generated index **4** of 64.

  **This reprices what G2 may demand of us.** vLLM does not reproduce its own
  greedy continuation across two of its own attention backends on this model, and
  its acceptance moves 6 points between them. A bar requiring our engine to be
  4-of-4 token-exact against one arm would be a bar vLLM fails against itself on
  the same hardware. That is the ratified near-tie envelope arriving as a
  MEASUREMENT rather than as an inherited argument, and `SPEC-DFLASH` D6 is
  corroborated rather than merely cited.

- **O23 — a hook on `_generate_draft` must stand aside during CUDA graph
  capture, and the capture instrument that proves the hook ran must not be the
  hook itself.** Owner: this row's tooling, recorded because the next agent to
  instrument this oracle pays it again otherwise. Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314).

  Two failures, both found by the instrument's own precondition check rather
  than by reading:

  1. **The engine core is a SEPARATE PROCESS by default.** vLLM V1 spawns
     `EngineCore`, which re-imports vllm clean, so an in-process monkeypatch is
     never on the object that drafts. The capture generated 4 x 64 tokens, vLLM's
     counters reported 50 drafts / 350 draft tokens / 209 accepted, and the hook
     recorded ZERO blocks. `VLLM_ENABLE_V1_MULTIPROCESSING=0` is the switch, and
     the repair asserts the client class rather than trusting the variable.
  2. **`capture_model()` calls `_generate_draft`**, so a `.tolist()` in the hook
     is a device-to-host copy inside a CUDA graph capture and torch refuses it
     (`RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
     capture`), at `dflash/speculator.py:147`. The hook now delegates whenever
     `torch.cuda.is_current_stream_capturing()`.
  3. **AND `_generate_draft` IS THE WRONG SEAM ENTIRELY under the production
     configuration.** `DFlashSpeculator.propose` branches
     `if batch_desc.cg_mode == CUDAGraphMode.FULL: run_fullgraph(...) else:
     self._generate_draft(...)` and returns `self.draft_tokens[:num_reqs]` on
     both arms, so with FULL decode graphs — which is what a denominator must
     use — the replay never enters `_generate_draft` in Python at all. This is
     the third failure and the most instructive: the run asserted
     `ENGINE_CORE_CLIENT=InprocClient` AND `HOOK_ON_CLASS=traced`, resolved
     `TRITON_ATTN`, generated 4 x 64 tokens, vLLM counted 47 drafts / 329 draft
     tokens / 216 accepted — and the hook still recorded ZERO blocks. Both
     preconditions it checked were TRUE and the instrument was still blind. The
     hook moved to `propose`, below the branch.

  **THE REPAIRED INSTRUMENT IS NOW COMMITTED**
  ([#1562](https://github.com/mudler/vllm.cpp/issues/1562)). Every element this
  entry described as prose is code in the tree, and the paragraph above is now
  the ARGUMENT for the code rather than a substitute for it:

  | O23 element | Where it lives |
  |---|---|
  | the hook on `propose`, below the `cg_mode == FULL` branch | `tools/bench/dflash2_oracle_capture.py::DraftRecorder.install` |
  | `torch.cuda.is_current_stream_capturing()` delegation | the same, counted as `skipped_capture` |
  | the in-process client assertion, on the RESOLVED CLASS | `::_assert_inproc_client` |
  | the resolved-backend read-back off the built engine | `::resolve_attention_backend` |
  | the abort on zero, and the one-sided bound | `tools/bench/dflash2_speed_harness.py::hook_reasons` |
  | the refusal gate over all of it, on CPU, with no wheel | `tests/tools/test_dflash2_speed_harness.py` |
  | the lease-side procedure, with the marker written from `trap ... EXIT` | `scripts/dflash2-speed-gate.sh` |

  What is NOT recovered: `w6-relabel.py` and the W6 run log were lost with the
  lease and are recorded here as lost rather than owed, because nothing can
  reconstruct them. The committed harness is what makes the NEXT capture
  re-derivable; it cannot make the previous one so.

  A related residual the harness would explain and nothing currently does: the
  TRITON_ATTN golden's `hook_stats` reads
  `{propose_calls: 59, skipped_dummy: 1, skipped_capture: 0}` against 55 recorded
  blocks whose `call` ids run contiguously 3..57. `59 - 1 - 0 = 58`, so THREE
  calls are unaccounted for. Nothing in the gate read `hook_stats` at all until
  the W6 repair wave, which now bounds it one-sidedly
  (`sum(blocks) <= propose_calls - skipped`, because a lost record is possible
  and an invented one is not) and pins the residual at 3 so it cannot drift
  unnoticed.

  The reason this is `## Owed` and not just a note: EVERY one of the three
  presents as a verdict about the CODE. The first reads as "the draft is empty",
  the second as "the oracle cannot capture graphs with a DFlash2 speculator" —
  which is false, and would have been a serious mis-record about upstream — and
  the third as "the drafter proposed nothing", on a run where vLLM's own counters
  say it proposed 329 tokens. The general rule is the one `.agents/verification.md`
  already states and this row paid for three times in one wave: a precondition
  check bounds only the failure it names, and a passing one is not evidence that
  the instrument SAW anything. Only the ABORT ON ZERO caught all three.

- **O24 — the `at_end` boundary fixture is COMMITTED AND UNPROVEN, because its
  mutation was never run.** Owner: `SPEC-DFLASH2`. Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314).

  `our_blocks_all == oracle_blocks_all` rests on `Reconstructed::verified`, which
  turns on `len < out.size()`. W6's second fresh review mutated that `<` to `<=`
  and **the whole suite stayed green**, because the pre-existing `tail` case has
  no block starting at exactly `len == out.size()` — its unconsumed blocks start
  at len 9 against an 8-token output, so the two operators agree there and the
  boundary was ungated. The second repair wave added `at_end`, which puts a block
  at exactly `out.size()`: block 0 accepts both drafts so the cursor lands on 4
  against a 4-token output, and `edge.verified` should read 1 under `<` and 2
  under `<=`.

  **The mutation that would prove it was NOT TAKEN.** It was attempted ONCE, and
  that attempt was voided by a harness race (two concurrent instances; the
  second baselined an already-mutated file, so it read `match count: 0`); a
  clean retake was then DECLINED as unaffordable, because every cycle in this
  build tree rebuilds the whole 464-object library while the box ran at loadavg
  145 and 12 objects per ten minutes. Running against the library already on disk was REFUSED: it
  had been compiled from the M1 mutation's source, so it would have measured a
  tree that no longer existed — the stale-binary trap `.agents/verification.md`
  names. Guessing the outcome would have been worse than recording the gap.

  **One voided attempt and one declined retake is what `### W6's MUTATION SET`,
  `.agents/benchmark-record.md` and the repair wave's own commit body all say.**
  An earlier revision of this paragraph read "attempted twice", which counted the
  declined retake as an attempt and left this row asserting two different
  descriptions of the same event in three places.

  **AN OUT-OF-SUITE COMPILATION SHOWS THE FIXTURE DISCRIMINATES, and it does NOT
  discharge this.** The third fresh review could not afford the in-suite mutation
  either, so it extracted `ReconstructAcceptance` into a standalone translation
  unit and compiled it both ways. Reproduced independently on 2026-08-21 by the
  repair wave: the function body copied VERBATIM from
  `tests/parity/test_qwen38_dflash2_spec_decode.cpp:308-325` -- signature at
  `:308-309`, `return r;` at `:324`, closing brace at `:325` -- with the
  comparison parameterised as `-DOP`, driven on the committed fixture's own literals,
  `g++ -std=c++17 -O0` rc 0 both ways.

  | case | `<`, the shipped operator | `<=`, the M2 mutant |
  |---|---|---|
  | `main` | verified 4, total 4, `[2,0,1,1]` | verified 4, total 4, `[2,0,1,1]` |
  | `tail` | verified 4, total 4, `[2,0,1,1,0,0]` | verified 4, total 4, `[2,0,1,1,0,0]` |
  | **`at_end`** | **verified 1**, total 2, `[2,0]` | **verified 2**, total 2, `[2,0]` |
  | `shift` | verified 3, total 4, `[2,0,2]` | verified 3, total 4, `[2,0,2]` |

  A `diff` of the two runs is ONE line, the `at_end` row. So `at_end` is the only
  discriminating case in the fixture, and `CHECK(edge.verified == 1)` at
  `tests/parity/test_qwen38_dflash2_spec_decode.cpp:1067` DOES red under M2 -- in
  the copy. That answers the risk this owed item was really carrying, that the
  new case proves nothing: the fixture discriminates. It answers it ON A COPY,
  and the copy was taken VERBATIM from `:308-325`, so the drift the next
  paragraph names as undetectable in general is nil HERE in fact rather than
  merely bounded. What the compilation cannot answer is whether the BUILT SUITE
  reaches the case, and that is a different question.

  **M2 STAYS NOT TAKEN.** What that compilation measures is a COPY of the
  function under a compiler, not the committed test binary: it does not exercise
  doctest's reporting, it cannot show that the built suite reaches the case, and
  a copy that had silently drifted from its original would read exactly the same.
  Those are the things an in-suite mutation is run for, so what is owed below
  stands unchanged.

  So the arithmetic case is gated and the BOUNDARY case is present but unproven.
  What is owed is one mutation on a quiet box: flip `<` to `<=` at
  `tests/parity/test_qwen38_dflash2_spec_decode.cpp`, rebuild
  `test_qwen38_dflash2_spec_decode` clean, and confirm `edge.verified == 1` reds.
  If it does NOT red, the fixture does not span the boundary and LOW-D is not
  repaired.

- **O25 — the DFlash2 startup notice prints TWICE on every draft load.** Owner:
  `SPEC-DFLASH2`. Issue
  [#1607](https://github.com/mudler/vllm.cpp/issues/1607).

  `CheckDflash2DraftArm` (`src/vllm/entrypoints/model_loader.cpp:502`) ends in an
  unconditional `std::cerr <<` with no once-flag, and the loader reaches it twice
  on one load of one `EngineParams`: directly from `FromModelDir` at `:1929`,
  and again from `ResolveSpecConfig` at `:1206`, which the `LoadedEngine`
  constructor runs in its member initializer at `:1538` on all three
  `new LoadedEngine(...)` returns (`:2172`, `:2341`, `:2359`). The server, the C
  ABI and the bench client therefore each print the paragraph twice, on the
  safetensors arm and the GGUF arm alike. The tree already states that the
  resolution re-runs (`model_loader.cpp:2313`, `:871-874`); what nothing stated
  is that the notice re-runs with it.

  **ESTABLISHED STATICALLY, by reading the call graph rather than by executing
  it.** Confirming it at runtime needs a DFlash2 checkpoint and a rebuild of the
  whole 464-object library, which the wave that found it was barred from paying
  for, so the mechanism is a reading and not a measurement.

  **WHAT RE-RUNS IS THE WHOLE CHECK, not only its trailing `std::cerr`, and the
  `#1607` index row's "nothing is loaded twice" is too strong for that.** On the
  GGUF arm the second reach is a second `vllm::GgufFile::Open(resolved)`
  (`src/vllm/entrypoints/model_loader.cpp:517`) and a second `IsDflash2Gguf`
  scan; on the safetensors arm it is a second `ReadDflashDraftArchitectures`
  read of the draft's `config.json` (`:531`). What is NOT duplicated is WEIGHTS,
  which is plainly what the row meant, and neither re-read is expensive next to
  the 51.75 GiB target load. The row is not edited, for the reason
  `## Upstream chain` gives for the `#1538` row: `.agents/issue-index.md` is
  append-only and a landed row is never edited. THIS ITEM IS THE AUTHORITY on
  what re-runs, and the index row is not.

  `docs/USAGE.md` called it a "one-time notice" and now describes the two prints,
  so the shipped documentation is no longer wrong about the behaviour while this
  is open. The FIX is deliberately NOT taken here: it changes the production
  loader and needs its own red-first test and its own fresh review. What would
  close it is one owner for the notice -- a once-flag inside
  `CheckDflash2DraftArm`, or dropping the constructor's call site, noting that
  `FromModelDir`'s exists on purpose so a misclassified draft is caught before a
  51.75 GiB target is mapped -- plus a test that loads a DFlash2 draft through a
  production entry point and asserts the paragraph occurs exactly once.

- **O26 — the MEASUREMENT IS TAKEN. `0.8016987337853048` on
  `output_throughput_tok_s`, `RECORDED, no floor declared`.** Owner:
  `SPEC-DFLASH2`. Issue
  [#1562](https://github.com/mudler/vllm.cpp/issues/1562).

  **THE HEADLINE OF THIS ENTRY WAS "the harness is committed and the MEASUREMENT
  is not taken" until 2026-08-22.** The run below happened on `dgx:gpu0` under
  `rc` job `ec9cf6cd-0aaf-4323-806d-6a12da2bd08f` at `d25730fbb`, exited
  `GATE_RC=0`, and emitted 13.051 against 16.27918250335551 tok/s. **That is a
  RECORDED RATIO and not a pass**: no floor was ever declared for this axis, and
  three of the four axes stay `NOT MEASURED`. `## Now`'s SPEED section carries
  the caveats, and [`.agents/benchmark-record.md`](../benchmark-record.md)
  carries the full entry with the command that derives each figure. **Residuals
  1 and 3 below are DISCHARGED TWICE OVER by that run**, which is recorded in
  each of them; residual 2's count of measured axes is now a count of what ran
  rather than of what the code does; residuals 4 and 5 stand unchanged. The
  procedure below stays as the RECIPE, because a repeat run needs it.

  What landed 2026-08-21 is the instrument and every refusal around it, gated on
  CPU with no GPU and no wheel — by the two commands below, and since O27 by
  `ctest --test-dir build -R test_serve_low_tools` and
  `scripts/agent-preflight.sh` as well:

  ```sh
  scripts/dflash2-speed-gate.sh --self-check     # syntax only, touches no GPU
  python3 -m unittest tests.tools.test_dflash2_speed_harness
  ```

  On a leased `dgx:gpu0`, the whole procedure end to end. Every flag below is
  REQUIRED: the two artifact lists are separate because the oracle loads an HF
  checkpoint and we load GGUF, so one list cannot identify both; and
  `--our-speculative-config` is required because our arm would otherwise run a
  plain decode against a drafting oracle and still fingerprint-match its k.

  **Provision the container first.** No CUDA toolkit is preinstalled, and two
  packages beyond the metapackage are required: `cuda-libraries-dev-13-0`,
  because `flashinfer.topk` JIT-compiles a `topk.cu` that includes
  `<curand.h>` and `cuda-toolkit-13-0` omits it, and `python3-dev`, without
  which Triton's driver JIT fails and reports
  `Model architectures ['Qwen3_5ForConditionalGeneration'] failed to be inspected`.
  The first of those killed a leg INSIDE `profile_run`, after a 12-minute model
  load. `.agents/environment.md` carries both
  ([#1660](https://github.com/mudler/vllm.cpp/issues/1660)).

  ```sh
  rc run --device dgx:gpu0 -- scripts/dflash2-speed-gate.sh \
      --evidence /workspace/evidence/dflash2-speed \
      --target /workspace/ckpt/qwen3.8-27b-hf \
      --draft /workspace/dflash2/draft-st \
      --oracle-commit <the merged head #1561 selects> \
      --oracle-build-recipe "<the exact pip line at that head>" \
      --attention-backend TRITON_ATTN \
      --artifact target=<path>=<sha256> --artifact draft=<path>=<sha256> \
      --our-binary /workspace/build/bin/vllm-cli \
      --our-model <our target> \
      --our-artifact our_target=<path>=<sha256> \
      --our-artifact our_draft=<path>=<sha256> \
      --our-speculative-config '{"method":"dflash","model":"<our drafter>","num_speculative_tokens":7}' \
      --our-build-recipe "<the exact cmake line>" \
      --repeat 5 \
      --num-speculative-tokens 7
  ```

  `--num-speculative-tokens` must equal the k inside `--our-speculative-config`.
  It goes to the oracle arm as the value it runs at and to our arm as a
  cross-check against the k the binary actually read, so a mismatch is a named
  refusal rather than a workload-fingerprint disagreement discovered by
  `summarize` after both arms have run.

  **Four residuals, each stated because a harness that hides one is worse than
  none:**

  1. **`BACKEND_PROBES` was UNVERIFIED against the beyond-pin wheel. DISCHARGED
     2026-08-22, by a leased run that MISSED on every probe and named the
     repair.** The prediction this entry made held exactly: all three walks
     raised `AttributeError: 'GPUModelRunner' object has no attribute
     'attn_backend'`, the refusal was loud, listed every probe it tried, took no
     fallback and invented no plausible label, and the repair was one entry.
     Two walks were then measured against the live engine and both returned
     `TRITON_ATTN`: `llm_engine.vllm_config.attention_config.backend` and
     `llm_engine.engine_core.engine_core.vllm_config.attention_config.backend`.
     Both are now first in `BACKEND_PROBES`; the three retired walks are KEPT
     after them, because the list is ordered, an older wheel still answers on
     them, and a walk that resolves nothing costs one `AttributeError`
     ([#1658](https://github.com/mudler/vllm.cpp/issues/1658)).

     **DISCHARGED A SECOND TIME, by the run that took the ratio.** The repaired
     list was exercised against the same beyond-pin wheel on 2026-08-22 under
     job `ec9cf6cd` and RESOLVED: `evidence/vllm-arm.json` records
     `attention_backend: TRITON_ATTN` with
     `attention_backend_source: read_back_from_engine` and
     `attention_backend_probe: llm_engine.vllm_config.attention_config.backend`,
     and the group walk returned the census below off
     `...model_runner.attn_groups`. So the first discharge measured that the
     probes MISSED and named the repair; this one measures that the repair
     ANSWERS on a run that also produced a number.

     **The same run also showed that ONE SCALAR under-describes this model.**
     `...model_runner.attn_groups` resolved three backends at once:
     `GDNAttentionBackend` over 48 `linear_attn` layers in 10 groups,
     `TritonAttentionBackend` over 16 `self_attn.attn` layers in 4 groups at
     every 4th index 3-63, and `FlashAttentionBackend` over the DFlash2
     draft's five sliding-window layers (`model.layers.64-68`) in one group.
     Those three counts are re-derivable from the run's own
     `c-probe-result.json` under `candidate_walks`, and this entry read 30
     for the GDN arm until 2026-08-22: 30 is the shape of the TEST STAND-IN,
     never a measured count
     ([#1666](https://github.com/mudler/vllm.cpp/issues/1666)).
     BOTH are recorded now and only the SCALAR is gated.
     The scalar is what the run declared, what `attention_backend_reasons`
     compares against, and what `test_qwen38_dflash2_spec_decode.cpp` reads off a
     golden; the map is what actually RAN. The map is NOT gated because the two
     spellings are not comparable -- the scalar is an enum name (`TRITON_ATTN`)
     and the map holds class names (`TritonAttentionBackend`) -- so a checker
     that equated them would refuse every correct run. A missed group walk is a
     named `miss` and never an empty map, because an empty map reads as "one
     backend over every layer", which is the false claim the field exists to
     prevent.

     **NOT DISCHARGED, and now ESCALATED because it touches a DENOMINATOR.**
     The run observed the draft's five layers resolving `FlashAttentionBackend`
     and generating tokens, while `FA-CONSTRAINT.txt` records `FA_USABLE=0` on
     sm_12x from #1456. #1456 needs a re-read before that constraint is quoted
     again. The 2026-08-22 ratio run reproduced it — the same five layers,
     `model.layers.64-68.self_attn.attn`, in the engine that became the
     denominator — so the question is no longer only about a constraint document:
     if those layers run FlashAttention, the denominator is not purely the
     `TRITON_ATTN` the row declares and the ratio carries a caveat nothing in the
     artifact states. Filed as its own issue for that reason,
     [#1685](https://github.com/mudler/vllm.cpp/issues/1685), and still tracked
     alongside #1658.
  2. **ONE OF THE FOUR AXES IS MEASURED, AND ON 2026-08-22 IT WAS.**
     `output_throughput_tok_s` came back at 13.051 ours against
     16.27918250335551 vLLM, each the median of sixteen warm legs; the other
     three read `NOT MEASURED` in the emitted artifact, exactly as this entry
     predicted. `output_throughput_tok_s` is measured on both
     arms: the oracle arm times `llm.generate` with `time.perf_counter()` and our
     arm reads `vllm-cli`'s own `tok_s`, and BOTH fold through the one shared
     `dflash2_speed_harness.fold_legs`, so the ratio is between two medians of
     the same statistic over warm legs of the same count.

     `ttft_ms` and `peak_device_bytes` stay `NOT MEASURED` because `examples/cli`
     reports neither. `tpot_ms` stays `NOT MEASURED` for a different and
     deliberate reason: wall time over completion tokens is
     `output_throughput_tok_s` inverted, so emitting it would fill an axis with
     the axis above it and hide that per-token latency, which excludes prefill,
     was never measured at all. THREE OF FOUR ARE OPEN GAPS and the result object
     says so in the field a reader looks at first.

     **This entry previously said two of three axes were open. That was wrong in
     the direction that flatters the harness.** As first committed the oracle arm
     imported no `time`, timed nothing, and built `metrics` by filtering
     `llm.get_metrics()` to names containing `spec_decode` -- so it shared NO key
     with our arm's `output_throughput_tok_s` and `build_speed_result` rendered
     all four axes `NOT MEASURED`, the vLLM denominator's throughput included.
     The harness could not emit a single ratio.

     **AND THEN "one of four" WAS ITSELF FALSE FROM `ad108f2fd` TO `747e62f4b`:
     over that range ZERO axes were measurable, because the oracle arm could not
     COMPLETE.** `capture()` resolved the read-back probe and did not pass it to
     `attention_backend_reasons`, which appends "no read-back probe was
     recorded" whenever `probe` is None -- so `require_no_reasons` raised on
     every run of the arm, after `LLM(...)` had loaded 51.75 GiB, on a lease.
     That is the failure O23 exists to prevent, and it was reachable by no test:
     `capture()` and `resolve_attention_backend` were executed by nothing, and
     the case that claimed to bind the read-back was a `source.count(...)` text
     grep. Found by the fresh review of `747e62f4b`, which executed the
     committed module against a stand-in wheel and got the refusal.

     Repaired here, one call site plus the tests that drive the arm end to end:
     `CaptureRunTest` runs `dflash2_oracle_capture.main()` against a stand-in
     `vllm`/`torch` and binds the two guarantees a green leased run would never
     announce it had lost -- the timed span excludes the model load, and the
     tokens counted are the ones GENERATED. Both mutations that make the number
     WRONG rather than absent (`completion = len(token_ids) +
     len(prompt_token_ids)`, and `elapsed = 1.0` with no `perf_counter`) were
     green before this change and are red after it. The count now stated as one
     of four is a count of what the code does.
  3. **The ANCHOR WALK was unverified, on the same footing as
     `BACKEND_PROBES`. DISCHARGED 2026-08-22 by the same leased run: the walk
     RESOLVED.** 10 of 10 recorded blocks carried an anchor and the capture
     reported `anchor_misses: 0`, so the consumer pairs on the anchor rather
     than falling back to ordinal pairing. **DISCHARGED A SECOND TIME by the
     2026-08-22 ratio run**, over a much larger population: `vllm-arm.json`
     records `hook_stats: {anchor_misses: 0, propose_calls: 355,
     skipped_capture: 0, skipped_dummy: 285}`, and `355 - 285 - 0 = 70` recorded
     blocks across the four records, which is exactly the block count the
     capture emitted (17 + 12 + 14 + 27). So the walk resolved on every
     non-skipped call and missed nothing, across exactly 7x the first discharge's
     population of 10 blocks. The rest of this entry stands as the design it
     describes.

     The capture emits the golden's own shape -- `records[i].blocks` with
     `num_blocks`, each block carrying `call`, `req_row`, `anchor` and `drafts`,
     beside the top-level keys `GOLDEN_TOP_LEVEL_KEYS` names -- because a capture
     whose output `tests/parity/test_qwen38_dflash2_spec_decode.cpp` cannot load
     could not become the golden this run exists to produce, and the FLAT
     top-level `blocks` list it first emitted could not. The anchor itself is
     read through `input_buffers.input_ids[_anchor_indices]`, the walk the
     oracle's own `_generate_draft` uses, and a miss is COUNTED rather than
     raised: a stopped 51.75 GiB run is the wrong answer to a moved attribute.
     `hook_reasons` refuses a capture that missed EVERY anchor, because without
     an anchor the consumer falls back to ordinal pairing, which compares blocks
     that started at different positions and reports the difference as a draft
     defect.
  4. **Inside a lease the SM clock can be SAMPLED and NOT PINNED** (`LGC_RC=4`
     even as root, #1354), so a pairing may be refused on within-run spread with
     no lever to fix it. The harness does not attempt `-lgc` and does not pretend
     the window was pinned.
  5. **THE INSTRUMENT PERTURBS THE DENOMINATOR, AND THE PERTURBATION IS
     UNMEASURED.** Two things cost the oracle arm and cost our arm nothing, so
     both flatter our ratio and both are conservative -- which is why they are
     written down rather than left out. `DraftRecorder.traced` stays installed
     for the WARM legs, because uninstalling it mid-run would change the object
     the timed repetitions call; on a warm leg it costs TWO
     `torch.cuda.is_current_stream_capturing()` calls per `propose` -- one on
     entry and one after the delegate, the `or` on the second not
     short-circuiting a `False` entry, which is every timed leg -- plus the
     counter increments. Only the anchor read and the `.tolist()` are behind
     `self.active` and skipped there.
     And `disable_log_stats=False` keeps vLLM's stat logger on so
     `llm.get_metrics()` can report the `spec_decode` counters the golden
     carries; a production serve leaves it on too, so that one is the
     denominator's own configuration rather than an instrument. Bounding the
     first needs a `--repeat` A/B with the hook uninstalled, and that needs the
     lease. Recorded at `tools/bench/dflash2_oracle_capture.py`, beside the
     `LLM(...)` call it applies to.

     The related limit is now CLOSED rather than owed: the gate exposed no
     `--num-speculative-tokens`, so it could run at no k but the capture's
     default of 7 -- any other k in `--our-speculative-config` made the two
     workload fingerprints disagree and `summarize` refused. A refusal is the
     right failure and it was still a gate that could not sweep the one knob
     this row's acceptance length turns on. `scripts/dflash2-speed-gate.sh` now
     threads the flag to BOTH arms, source of truth on the oracle's side and a
     cross-check against the config the binary read on ours.

- **O27 — [#1646](https://github.com/mudler/vllm.cpp/issues/1646)'s central
  claim is FALSE, and its index row cannot be edited.** Owner: `SPEC-DFLASH2`.
  Issue [#1648](https://github.com/mudler/vllm.cpp/issues/1648).

  #1646 landed beside O26 saying that `tests/tools/`'s 351 cases across 20
  suites had "no workflow, no CTest registration and no
  `scripts/agent-preflight.sh` line" running them. **The CTest registration
  exists and CI runs it on every pull request.** `tests/CMakeLists.txt:12-16`
  registers `test_serve_low_tools`, which runs exactly
  `python3 -m unittest discover -s tests/tools -p "test_*.py"` with `PYTHONPATH`
  set, and has since `e58858a91`; `CMakeLists.txt` makes it live under
  `if(VLLM_CPP_BUILD_TESTS)` and `.github/workflows/ci.yml` runs
  `ctest --test-dir build --output-on-failure` inside `build-test-cpu`. No line
  number is given for either, because the #1648 index row gave one and it was
  stale on arrival: it cited `ci.yml:1057`, and the very commit that wrote the
  row also added 11 lines to `ci.yml` above that point, so `ctest --test-dir
  build` sat at `:1068` in the tree the row shipped in and has moved again
  since. An anchor that moves inside its own commit is not an anchor.

  **THREE NUMBERS IN THE #1648 ROW AND IN THIS ENTRY'S FIRST DRAFT ARE WRONG,
  and the row is append-only, so this entry is the correction.** Re-measured on
  a clean tree at each named revision:

  | Claim as written | Measured | Command |
  |---|---|---|
  | "`Ran 414 tests / OK` ... on the branch head" | 414 is the count at `9f7fac052`, the PRE-repair head. At `747e62f4b`, the head the row shipped on, it is **455** | `python3 -m unittest discover -s tests/tools -t . -p "test_*.py"` at each revision |
  | "the ledger reads 35/35 on two rows and 164/164 on eleven" | 35/35 on two rows is right; 164/164 is on **13** | `grep -c 'tools 164/164' .agents/parity-ledger.md` |
  | "`.github/workflows/ci.yml:1057`" | `1068` at `747e62f4b`, moved by the same commit's own `ci.yml` edit | `git show 747e62f4b:.github/workflows/ci.yml \| grep -n 'ctest --test-dir build --output'` |

  The substance of #1648 is unaffected: the CTest registration exists, CI has
  been running it, and the correction stands. What was wrong was the arithmetic
  around it, in the direction that made the correction look better checked than
  it was.

  The claim came from grepping `tests.tools`, the DOTTED module path, while
  CMake and the workflow spell it `tests/tools`. A null grep proves the search
  terms wrong, never that the thing is absent.

  What follows: the `.agents/parity-ledger.md` "all tools" citations and
  `.agents/upstream-sync.md:38` were citing a LIVE suite; the narrow gap was
  real and its fix stands, because preflight ran none of them; and the DISCOVERY
  mechanism is correct and unaffected. Two further errors in the same row --
  "pinned at 3", which pins nothing because the bound holds for any residual at
  or above 0, and "all tools 34/34 on five rows", which does not occur -- are
  recorded in #1648.

  **`.agents/issue-index.md` is append-only, so the #1646 row STANDS AS
  WRITTEN.** This entry and #1648 are the authority over it, exactly as this
  campaign handled the #1538 row. The prose in `.github/workflows/ci.yml` and
  `scripts/agent-preflight.sh` is corrected in the change that files #1648, and
  the three counts above are corrected here. NOTHING IS OWED beyond that: the
  mechanism stays, only its justification and its arithmetic were wrong.

- **O28 — the DSpark lane still refuses a quantized target head, and the true-W4A4
  head arm is unimplemented in both lanes.**
  [#1628](https://github.com/mudler/vllm.cpp/issues/1628), 2026-08-21. Two arms
  are named-and-refused rather than built.

  **NOT the O28 that [#1657](https://github.com/mudler/vllm.cpp/issues/1657),
  [#1658](https://github.com/mudler/vllm.cpp/issues/1658),
  [#1659](https://github.com/mudler/vllm.cpp/issues/1659) and
  [#1660](https://github.com/mudler/vllm.cpp/issues/1660) name.** Those index
  rows were written on a branch that authored its entry as O28 while `main`
  landed this one first; the rows are append-only and cannot be corrected, so
  the redirect lives here. They mean **O30**.

  `LoadDsparkDraft` passes `head_fp4 = nullptr` to `SharedHeadSource::LoadInto`,
  so a DSpark draft off an NVFP4 target still dies at
  `dflash: target tensor lm_head.weight is not BF16 (got U8)`. The reason is
  structural and not an oversight: `Qwen3DSparkWeights::backbone` holds ONE bf16
  `lm_head` owner, and giving it a second owner is a change to the DSpark
  backbone and its forward, which is `SPEC-DSPARK`'s surface and not this one's.
  The argument is `nullptr`-and-not-defaulted for the reason W3 gave for
  `head_was_quantized`: dropping it is a compile error rather than a silent
  behaviour change.

  A head that is NVFP4 with the ACTIVATION divisor in force (`VT_MODELOPT_W4A4=1`)
  refuses BY NAME in `DflashLogitsF32D`, because the draft's shared-head GEMM is
  the W4A16 dispatcher and the target's own head is W4A16 under both spellings
  unless that variable puts the divisor back (vLLM's
  `ModelOptNvFp4W4A16LinearMethod` DELETES it, `modelopt.py:1365`). Routing the
  draft into the fp4-activation GEMM while the target uses another one is the same
  silent-wrong D12 is about, so it refuses instead.

  **There was a SECOND copy of that refusal, at load, and the mutation pass is
  what removed it.** A startup refusal reads better and was written first. Then
  disabling the forward guard left the suite GREEN, because the W4A16 dispatcher
  refuses the same weight one layer down with its own message and the case was
  asserting on a substring both messages carry. Two guards for one rule that no
  test can tell apart is the "two descriptions of one rule" failure AGENTS.md
  `## Changing the rules or a checker` names, so one survives -- the one a gate
  reaches -- and the case now asserts on what only it can say, the variable to
  unset. The refusal arriving at the first propose rather than at load is the
  polarity D10 already set for this lane.

- **O29 — the packed shared head is gated on CPU only; the CUDA arm and the real
  checkpoint are OWED a measurement.**
  **Not the entry the `#1657`-`#1660` index rows mean either; see O30.** Those
  rows say O28, this file's O28 and O29 both belong to the DSpark lane, and the
  entry they were written for is O30.
  [#1628](https://github.com/mudler/vllm.cpp/issues/1628), 2026-08-21. The
  implementer had no GPU (`BENCH-QWEN38-27B-SOTA` held the fleet), so **THIS
  CHANGE'S OWN gates are all CPU**. That is a statement about this row and not
  about the lane: W6 took device gates for `SPEC-DFLASH2` on `dgx:gpu0` the same
  day (`## Now`), and the SGLang observation in `## Upstream chain` is itself a
  device measurement. What no device has seen is the packed head. Three things
  are therefore unmeasured and are claimed nowhere:

  1. The equality the CPU gate measures is measured on the CPU arm of the W4A16
     dispatcher (host dequant + bf16 `vt::Matmul`). On CUDA both sides take the
     Marlin NVFP4 GEMM instead, and **they are not the same function**. The
     target's logits take `MatmulNvfp4F32D` (`qwen3_5.cpp:3106`, reached from
     `:3139`) while the draft takes `dense_nvfp4::MatmulNvfp4W4A16D`, and the two
     `MatmulNvfp4MarlinD` bodies below them (`qwen3_5.cpp:2849-2903` vs
     `dense_nvfp4_gemm.h:505-560`) hold SEPARATE function-local `static void* ws`
     workspaces, thread `w.group_size`/`w.is_mxfp4` in one and hardcode `16`/
     `false` in the other, and differ on a `MutableW4A16Stats()` counter. For a
     HEAD the last two collapse: `LoadCtNvfp4Raw` and `LoadNvfp4AnyNaming` set
     neither field, so both stay at the `Nvfp4Weight` defaults `group_size = 16`
     and `is_mxfp4 = false`, which is what `vt::MoeMarlinArgs` already defaults
     to and what qwen3_5.cpp hardcodes, and the counter is observational. **THE
     ONE THING THE CUDA RUN MUST CHECK IS THE WORKSPACE-ZERO POLICY**: qwen3_5.cpp
     `Memset`s the Marlin lock workspace before EVERY call, and the shared copy
     zeroes it ONCE on a documented kernel-self-reset invariant
     (`dense_nvfp4_gemm.h:495-500`). Two separate workspaces under two policies
     is the thing that could make the two paths disagree on a device, and no
     reading settles it -- the equality is what this row is gated on.
  2. `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a2` has never been loaded
     with a DFlash2 draft attached. The load refusal that #1628 reports is gone
     in this tree by construction; that it now LOADS, drafts, and holds the
     `19 x 23 -> 437` canary is unrun.
  3. Acceptance. The whole claim of DFlash2 is acceptance, and this row moves the
     head the selector reads. `## Gates` owns the acceptance gate and none of it
     is taken here.

  The load-time double residency belongs to the same lease. `LoadDenseLmHead`
  sets `keep_dequant_b = true` on the head it loads, and the draft now holds its
  own `Nvfp4Weight` beside the target's, so a backend with NO fp4 GEMM keeps TWO
  dequantized bf16 head operands rather than one. On CUDA neither is built. The
  policy is deliberately the target's own and not a second opinion; the measured
  cost on a non-CUDA backend is owed with the rest.

- **O30 — the committed gate could NOT produce a number, and four issues say
  why.**

  **This entry was authored as O28 and renumbered on the merge with
  `origin/main`, which landed its own O28 and O29 first.** The append-only
  `#1657`, `#1658`, `#1659` and `#1660` rows in `.agents/issue-index.md` still
  say `## Owed` O28, because they were written before that merge and an index
  row is never edited. This is the entry they mean; O28 above is the DSpark
  lane and is unrelated to them. Owner: `SPEC-DFLASH2`. Issues
  [#1657](https://github.com/mudler/vllm.cpp/issues/1657),
  [#1658](https://github.com/mudler/vllm.cpp/issues/1658),
  [#1659](https://github.com/mudler/vllm.cpp/issues/1659),
  [#1660](https://github.com/mudler/vllm.cpp/issues/1660).

  O26 asked for the run. The run happened on `dgx:gpu0` on 2026-08-22 at
  `bed3feae6`, over leases `11cee02a`, `52ac5673` and `a03f34e4`, and it
  measured the INSTRUMENT rather than the engines: the gate as committed cannot
  emit a ratio by any path. Two residuals of O26 were discharged by it and are
  marked so above; four defects were filed. Three of them block a measurement
  independently, so fixing any one alone still yields nothing.

  | Issue | What it stopped | Repaired here |
  |---|---|---|
  | #1657 | the clock summary must describe the arm AND exist before it | the ARM owns its window: it runs, its sampler stops, the summary is written, and only then is it read and judged |
  | #1658 | all three `BACKEND_PROBES` walks raised `AttributeError` | the two measured walks lead the list, and the per-group map is recorded beside the scalar |
  | #1659 | `LLM(...)` took no backend kwarg, so `resolved == declared` was unreachable | the declared backend is passed, over the spellings `ATTENTION_BACKEND_KWARGS` names |
  | #1660 | `--lease-id` defaulted to `$RC_LEASE_ID`, which this fleet does not export | the default is `$RC_JOB_ID`, and the recipe above names the two packages a lease needs |

  **Why #1657 was invisible to a green suite.** `setUp` pre-wrote `clock.json`,
  so every case received a summary that already existed and no case drove the
  DRIVER's ordering. The suite could not fail the way the lease failed. Nothing
  is pre-written now, a stub sampler writes the summary on STOP, and a
  `.running` marker makes "the window is open right now" observable -- the
  samples file survives the sampler, so reading it would have answered yes to a
  window that had already closed.

  **The design choice, stated because a reviewer must be able to reject it.**
  The clock is a precondition of the MEASUREMENT, not of the arm's execution, so
  a run whose clock later proves unusable is discarded at judgement time rather
  than prevented from starting. Two consequences are deliberate. The arm record
  is WRITTEN BEFORE the verdict, because a leased arm costs about two hours and
  discarding its evidence makes the next run pay the same lease to see the same
  thing; nothing quotable is emitted, since the refusal precedes the `print` and
  `build_speed_result` refuses the same record again through `clock_pairing`.
  And the oracle arm's window opens AFTER `LLM(...)` has loaded, so the load is
  outside the samples `clock_reasons` floors at 50% busy.

  **That guarantee was FALSE on the one branch it was written for, and a fresh
  review found it.** It held only where the sampler had written a summary.
  `gpu_clock_state.run_sampler` builds the record BEFORE it writes it, and
  `build_clock_record` refuses an entirely idle window -- `every one of N clock
  samples was idle`, the exact string this run met -- and refuses a mid-window
  field change, while `query_once` refuses a failed `nvidia-smi`. On any of
  those the sampler exits 2 with NO summary, `ClockWindow.close(read=True)`
  raised, the exception left the `with` block and `capture()`, and `main()`
  never reached `write_json_atomic`. The whole golden went with it -- records,
  blocks, token ids, legs -- which is the provenance O26 needs. The
  `stop_timeout` kill path lost it identically, and both existing
  "unusable window" cases fed a summary the sampler DID write, so neither ever
  reached this branch. `ClockWindow.__exit__` now keeps the failure in
  `close_error`, leaves `record` None, and the driver refuses on it through
  `clock_state_reasons` with the sampler's own words appended. Three cases drive
  it: refuse-at-stop on both arms and the kill path on the oracle arm.

  **The overwrite refusal cost a full model load, and no longer does.**
  `gpu_clock_state` refuses to overwrite either file it writes, and since the
  sampler moved inside the arm that refusal was first evaluated when the ARM
  opened its window -- for the oracle arm after `LLM(...)`, 702 s on this box.
  `scripts/dflash2-speed-gate.sh` does `mkdir -p "${EVIDENCE}"` and asked
  nothing. `clock_evidence_reasons` now runs both `path.exists()` checks inside
  each arm's `precheck`, the gate threads `--clock-summary` into both
  `--precheck-only` invocations, and a rerun into a used evidence directory
  stops before the lease does any work.

  **Two sizes describe this one load and they are not the same measurement.**
  `out-o26c/c-r1-oracle.log` on the share carries both, from the same
  `LLM(...)`: `weight_utils.py:858` reports `Checkpoint size: 51.75 GiB` for the
  target and `3.58 GiB` for the draft, both read off CIFS; `model_runner.py:385`
  reports `Model loading took 54.87 GiB memory and 702.391374 seconds`, which is
  what the finished load HOLDS. So 51.75 GiB is the target checkpoint on disk
  and 54.87 GiB is resident memory. Neither figure was wrong; the two sites that
  quoted them said "load" for both, and each now names what it measures.

  **THREE THINGS STAY OWED, and the next leased run finds them.**

  1. **The kwarg SPELLING is UNVERIFIED**, on the same footing `BACKEND_PROBES`
     was on before this run. `attention_config={"backend": ...}` is tried first
     because the measured read-back walk is `vllm_config.attention_config.backend`,
     `attention_backend=...` second. A wheel that takes NEITHER is a loud refusal
     naming both, raised by `EngineArgs` before anything loads, so the search
     costs no lease time; `--attention-backend-kwarg` pins the answer once it is
     known, with no code change. A spelling that is ACCEPTED and IGNORED is
     caught by the read-back, which is the refusal #1659 already describes.

     **The fall-through is `TypeError` ONLY, which is a second exposure in the
     same place.** A wheel that DECLARES `attention_config` and validates its
     shape raises something else -- a `ValueError`, a pydantic
     `ValidationError` -- and that propagates and stops the arm rather than
     trying `attention_backend`. It is named rather than repaired: the failure
     is loud, it is pre-load so it costs no lease time, and
     `--attention-backend-kwarg` is the lever. Catching more would be WORSE, in
     a way that costs a measurement: a wheel that rejects our VALUE would be
     retried under another spelling and the refusal would then name the wrong
     cause. Widen it when a second shape is measured, never in advance.
  2. **OUR ARM'S WINDOW SPANS FOUR MODEL LOADS.** `vllm-cli` is one process per
     prompt and loads inside the window, so ours cannot exclude a load the way
     the oracle arm can. If `clock_reasons` refuses it on the 50% busy floor,
     the lever is `vllm-cli` marking its own leg boundaries, which this row does
     not own. Recorded rather than worked around.

     **DISCHARGED BY O32, and `clock_reasons` did refuse it — at 18.37% busy.**
     `vllm-cli` marks its own leg boundaries now, and the record our arm is
     judged on is built from the samples inside its WARM leg spans. The window
     is still opened by the arm exactly as this entry's table says; what changed
     is which of its samples the record describes. Read O32 for the arithmetic,
     including the one attribution in this entry's own reasoning that was wrong.
  3. **`FA_USABLE=0` on sm_12x needs a re-read** before it is quoted again: the
     same run watched the draft's five layers resolve `FlashAttentionBackend`
     and generate tokens. Tracked in #1658.
  4. **THE GDN COUNT IN THIS ENTRY WAS THE TEST STAND-IN'S, and it read 30
     until 2026-08-22.** The run resolved 48 `linear_attn` layers in 10 groups,
     not 30 in 1. 30 was `_stand_in_attn_groups()`'s `range(30)`, generalised
     into a sentence that four files then carried as a measurement, on the row
     whose whole history is that failure class. Corrected in the four files and
     in the fixture, which now carries the measured 10/4/1 group shape and the
     real layer names, so the next reader lifts the census rather than the
     stub. The fifth copy is in the `#1658` index row and stays: that file is
     append-only. Re-derive from `candidate_walks[...attn_groups]` in the run's
     own `c-probe-result.json`, never from this paragraph.
     [#1666](https://github.com/mudler/vllm.cpp/issues/1666).

- **O31 — both arms judge the LEGS before the arm record reaches disk, so an
  ordinary refusal discards a lease.** Owner: `SPEC-DFLASH2`.
  [#1667](https://github.com/mudler/vllm.cpp/issues/1667), 2026-08-22, filed
  and deliberately NOT fixed in the flow that found it.

  `require_no_reasons(hook_reasons(...) + leg_reasons(...))` sits after the
  clock window and before the record is written, in `capture()` on the oracle
  arm and in `main()` on ours. One prompt returning 63 completion tokens
  instead of the asked 64 therefore takes `records`, `blocks`,
  `output_token_ids` and the O26 provenance with it, after about two hours of
  lease. Driving the existing `OurArmRunEntryPointTest` fixture at
  `completion = 63` prints `arm JSON on disk: ABSENT` against the control run's
  `PRESENT`.

  It is a separate defect from #1657 and not a regression of it. #1657 moved
  the CLOCK judgement after the write on exactly this reasoning, and both arms
  carry the comment saying so; that guarantee is scoped to the clock and says
  nothing about the leg and hook checks. `git log -S` places both call sites at
  `208559a79` (#1653), which predates #1657 and predates the branch that found
  it. The repair changes the refusal ordering of both arms and wants its own
  red-first test per arm, which is why it is owed rather than taken here.
- **O32 — DISCHARGES O30 ITEM 2 (authored as O28 item 2): our arm's clock window
  is the WORK now, not the process.** Owner: `SPEC-DFLASH2`.
  [#1671](https://github.com/mudler/vllm.cpp/issues/1671) fixed in flow,
  [#1673](https://github.com/mudler/vllm.cpp/issues/1673) filed and owed here.

  **THE NUMBERING, RESOLVED.** This entry was authored O32 against a `main` that
  carried neither O30 nor O31, on the reasoning that a gap is harmless and a
  collision is not. Both landed at `3e5d2f370` while this branch was open, so
  the list now reads O29, O30, O31, O32 with no gap and the id stands as
  authored. Nothing was renumbered on this merge, and nothing needed to be: the
  id has already collided twice on this row — O30's own second paragraph is the
  record of the second time, when an entry authored as O28 had to become O30 —
  and the append-only `.agents/issue-index.md` rows that name an old label
  cannot be edited, which is why a redirect sentence lives in the entry the
  reader lands on rather than in the row that pointed at it.

  **What refused, and why the refusal was right.** The repaired gate ran end to
  end on `dgx:gpu0` on 2026-08-22 over lease `9ee9f53a` at `04ed7b984`, and
  stopped at our arm's last precondition:

  ```
  REFUSED: DFlash2 our-arm capture REFUSED, 1 precondition(s) unmet:
    - clock: ours was idle for 2630 of 3222 SM-clock samples (18.37% busy, below
      the 50% floor); the retained window does not describe the measured work
  ```

  The arithmetic is the arm's own legs. The window was 3377 s. All 20 legs
  together were 1052.5 s, of which the four cold legs were 959.3 s and the
  sixteen warm legs — the only legs `fold_legs` folds — were **93.2 s**. About
  2325 s of the window sat inside no leg at all, because `vllm-cli` is one
  process per prompt and each one reads a 52 GiB checkpoint off CIFS before it
  can decode anything. **The window was about 3% warm generation.**

  **ONE ATTRIBUTION IN #1671'S OWN TABLE IS WRONG, and it is worth correcting
  because it points the repair at the wrong seconds.** That table labels the
  959.3 s of cold legs "the `vllm-cli` model loads". It cannot be: `main.cpp`
  times `vllm_complete` alone and `vllm_engine_load` runs BEFORE the repeat
  loop, so no leg of this arm has ever contained a load. The loads are the
  ~2325 s that sat inside no leg — which is why marking the legs is what removes
  them, and why the four cold legs at 240 s each are something else again: the
  first graph capture, the first KV allocation and the first touch of weights
  the loader had only mapped.

  **THE FLOOR DID NOT MOVE AND MUST NOT.** `MIN_BUSY_FRACTION` is still 0.5 and
  `MIN_BUSY_SAMPLES` is still 30. What moved is the WINDOW. A driver that can say
  when its work ran hands the spans to
  `gpu_clock_state.build_spanned_clock_record`, which delegates to
  `build_clock_record` — so the statistics, the idle accounting and the
  mid-window field check are the same code the unrestricted path runs, and
  `clock_reasons` judges the result unchanged.

  **The choice, stated so a reviewer can reject it.** #1671 offered two shapes:
  (a) leg-boundary markers, and (b) a resident engine across prompts, as the
  oracle has. **(a) is implemented, and the decisive argument is not the busy
  fraction.** It is that (b) cannot put the clock on the statistic at all. The
  number is a median over sixteen warm legs; a resident arm's window still spans
  a load and four discarded cold legs, so its record would still describe work
  the number excludes. Only spans remove that, and once there are spans the
  window is generation whether the engine is resident or not.

  **The busy fraction says the same thing less certainly, and the estimate is
  written with its uncertainty rather than as a figure.** The run retained 592
  busy samples of 3222 (18.37%), and about 2325 s of the window sat inside no leg
  — call it 581 s per load. Dropping three of the four loads shortens the window
  to roughly 1634 s against the same 592 busy samples, about **36%** busy: under
  the 50% floor, and refused. That estimate assumes the four cold legs cost with
  one load what they cost with four, which they would NOT — three of them would
  no longer pay a first touch — so the true figure is higher than 36% and this
  run cannot say by how much. An earlier draft of this entry read "roughly 13%",
  computed as warm generation over load plus warm generation, which drops the
  cold legs from the numerator while a busy sample does not care whether the leg
  it landed in was folded. It is corrected here rather than left standing,
  because a false count in this entry is the failure this row keeps having.

  **The window is the WARM legs, not all twenty.** `fold_legs` discards run 1 of
  each repetition group for a named cause, so a window that kept run 1's span
  would attribute the number to work the number excludes — the same defect this
  entry is about, three orders of magnitude smaller and correspondingly harder to
  see. `dflash2_speed_harness.is_warm_leg` is the ONE predicate both consumers
  read, so the number and the clock that qualifies it cannot come to describe
  different legs.

  **What was added, all of it a refusal.** No span at all is refused rather than
  widened back to the whole stream — that fallback would silently restore exactly
  the window this entry is about, under a record that claims to be spanned. A
  reversed span is refused. A span set that retained no sample is refused naming
  the stream it missed, because a driver whose spans miss the sampler's window is
  a driver defect and never a fast window. And a leg printed with no boundary
  marker is refused naming `examples/cli/main.cpp`, so a binary built before the
  marker cannot quietly drive this arm.

  **Where the marker comes from, and the one failure a source contract cannot
  see.** `examples/cli/main.cpp` prints a second line per leg,
  `run=%d/%d generate_start_unix=%.6f generate_end_unix=%.6f`, from
  `system_clock`; the timing line keeps `steady_clock`, because a DURATION and an
  INSTANT are different questions and only the instant can be lined up against
  another process's samples. `LegMarkerContractTest` reads the format string out
  of the source and asserts the harness's regex matches what it renders, which
  catches a rename. It cannot catch a wrong VALUE, and the wrong value here is
  total: `steady_clock`'s epoch on Linux is BOOT, so a mixup would place every
  span tens of days away from every sample and retain nothing.
  `CliMarkerRuntimeTest` therefore LINKS the production `main.cpp` against a stub
  libvllm, RUNS it, and asserts the printed instants land between a `time.time()`
  taken before and after the process. That case skips loudly with no C++ compiler
  on `PATH`, and says in its own words that a skip is not a pass.

  **THIS DID NOT MAKE THE GATE EMIT A NUMBER ON ITS OWN, and nothing here ever
  claimed it did. IT DOES NOW, BECAUSE #1657 IS BESIDE IT.** Authored alone,
  this entry left the oracle arm running FIRST and still unable to read a clock
  summary that does not exist until its sampler stops
  ([#1657](https://github.com/mudler/vllm.cpp/issues/1657)), so a leased run
  stopped there before our arm was reached. O30's repair landed at `3e5d2f370`
  and this branch was reconciled onto it, so both are on one tree: the oracle
  arm owns its window and opens it after `LLM(...)`, our arm owns its window and
  is judged on the samples inside its warm leg spans, and no refusal between the
  two remains that either entry knows of. What THIS entry removes is the LAST
  refusal, the one that only appears after both arms have run — so it is the one
  that costs a whole lease to discover, and the one O30 item 2 named as owed.

  **HOW THE TWO REPAIRS WERE RECONCILED, because both rewrote `main()`.** Each
  authored its own version of our arm's clock handling and the merge had to pick
  one shape rather than keep two. `ClockWindow` keeps the OUTER window: the arm
  starts the sampler, waits for its first sample, and stops it after the last
  leg, which is #1657's whole point and is what makes a finished stream exist at
  all. The spanning then reads `window.samples` — the file that sampler wrote —
  where the branch had read a `--clock-samples` path the shell filled in. **The
  two paths this entry authored are COLLAPSED INTO ONE.** `main()` briefly had a
  spanned path under `--clock-samples` and the old whole-window path without it;
  after the merge the spanned path is the only one, `--clock-samples` comes from
  the shared `add_clock_arguments` and names one file the arm both writes and
  reads, and the record the arm is judged on is always the warm-leg one. A
  second judged path is a second thing to keep true, and a run that took the
  wrong one would fail on the busy floor for a reason nobody could see in its
  argv. Two consequences follow and both are deliberate: reading the stream
  AFTER the `with` block is now strictly safer than the per-line flush this
  entry relied on, because `ClockWindow.close` has already signalled the sampler
  and waited for the process to exit; and a sampler that fails at STOP refuses
  the run even when the spanned record built cleanly from the raw stream, which
  is stricter than either side alone and keeps #1657's guarantee that the
  sampler's own words reach the operator.

  **ONE SENTENCE OF THE APPEND-ONLY `#1671` INDEX ROW IS SUPERSEDED HERE.** That
  row says `scripts/dflash2-speed-gate.sh` hands our arm "the stream
  `open_clock_window ours` writes". `open_clock_window` no longer exists: #1657
  removed the shell sampler entirely, and the gate now names
  `--clock-samples "${EVIDENCE}/clock-ours-samples.jsonl"`, which is the file
  the arm's OWN sampler writes and the arm then reads back. The flag and the
  path are unchanged; only who writes them is. The row cannot be edited, so the
  correction lives here.

  **STILL OWED, and filed.** The oracle arm's window is not restricted to its
  own warm legs, so the two arms' records describe different leg populations
  while `compare_clock_records` gates their median and mean offsets against each
  other at 1.0% each. **The reconciliation makes it fixable, which the
  as-authored text said it was not.** #1673 was filed against a tree where the
  oracle arm never saw a finished sample stream; `ClockWindow` gives it one, and
  `capture()` already times each leg, so the same `warm_leg_spans` and
  `build_spanned_clock_record` our arm calls now have somewhere to be called
  from on that side too. It stays OWED and is not taken here: it changes the
  denominator's clock record and wants its own red-first case.
  [#1673](https://github.com/mudler/vllm.cpp/issues/1673) carries it and names
  the two functions that close it — the same two our arm calls, so the arms end
  up folded by one rule rather than by two.

  **RECORDED AND NOT OWED: what (b) would still have bought.** Our arm pays four
  model loads and the oracle one. After this change neither the loads nor the
  cold legs are inside the number or inside the clock that qualifies it, and the
  fold populations already match — `fold_legs` discards run 1 of each repetition
  group on BOTH arms, so both fold 16 warm legs of 20. What remains is per-process
  state that our arm re-pays four times and the oracle once: page cache, allocator
  arenas, first-touch. Nothing measured here sizes that, and filing an unsized
  difference as a defect is the failure this row keeps having, so it is written
  down rather than filed.

- **O33 — the denominator is NOT vLLM's production configuration, and the
  constraint that justified substituting one is RETRACTED AT THE ARTIFACT.**
  Owner: `SPEC-DFLASH2` for this row's ratio, the developer for the declaration.
  [#1796](https://github.com/mudler/vllm.cpp/issues/1796) carries this entry and
  its evidence, [#1456](https://github.com/mudler/vllm.cpp/issues/1456) is the
  retracted premise, and [#1685](https://github.com/mudler/vllm.cpp/issues/1685)
  is the observation it explains.
  **NOTHING IS RE-MEASURED HERE AND NO DENOMINATOR IS SUBSTITUTED.**

  O22 records that a W6 RUN falsified #1456's conclusion. This entry records that
  the WHEEL ITSELF falsifies it, off-GPU, so the retraction no longer rests on
  reading a log that was lost with its lease. Both staged oracle wheels were read
  with `zipfile` and a fatbinary walk on the CPU dev box — no lease, no GPU, no
  CUDA toolkit:

  | wheel | module | fatbins | images per fatbin | arch |
  |---|---|---:|---|---:|
  | `0.1.dev1+g66e5414c6` (this row's oracle) | `_vllm_fa2_C.abi3.so` | 76 | ELF **and PTX** | 80 |
  | `0.1.dev1+g66e5414c6` | `_vllm_fa3_C.abi3.so` | 192 | ELF **and PTX** | 75 |
  | `0.1.dev1+g555967922` (the parity pin) | `_vllm_fa2_C.abi3.so` | 76 | ELF **and PTX** | 80 |
  | `0.1.dev1+g555967922` | `_vllm_fa3_C.abi3.so` | 192 | ELF **and PTX** | 75 |

  Every fatbinary carries a PTX image beside its SASS image. The first FA2 PTX
  payload is zstd, and it decompresses to `.version 9.0` / `.target sm_80` for
  `flash_fwd_hdim128_bf16_causal_sm80`. **That is the `+PTX` half of
  `FA2_ARCHS "8.0+PTX"`, and it is the mechanism by which the module CAN reach
  sm_121.** Be exact about what that buys: the artifact establishes a NECESSARY
  condition, that forward-JITtable code is shipped. That the JIT then ran is an
  inference from the PTX being there and a run selecting `FLASH_ATTN` and
  generating. #1456 read the SASS arch and concluded the module "cannot target
  sm_12x"; the arch reading is right, and the conclusion drops the PTX, which is
  enough to retract it.
  `cudaErrorUnsupportedPtxVersion` is raised when PTX ISA is NEWER than the
  driver, and `.version 9.0` under driver 580.173.02 is not that case. vLLM says
  the same thing in its own words: `FlashAttentionBackend.supports_compute_capability`
  returns `capability >= DeviceCapability(8, 0)` at
  `vllm/v1/attention/backends/flash_attn.py:251-252` in the staged wheel, so 12.1
  is a capability upstream declares supported.

  **WHAT THAT DOES TO THE RATIO.** #1456's body records the denominator decision
  in one sentence: the DFlash2 speed gate's denominator "will be vLLM pinned to
  `TRITON_ATTN`, by developer decision on 2026-08-20 ... it is NOT vLLM's default
  backend on this device, and any ratio taken against it must say so." AGENTS.md
  requires vLLM's PRODUCTION configuration as the denominator. On this box that
  configuration selects `FLASH_ATTN`, and the gate run's own log shows both paths
  in the same process: the forced path took `TRITON_ATTN` for the 27B target
  (`out-n1673b/m-gate.log:30`, `cuda.py:426`, the branch that honours an explicit
  request) and the auto path chose `FLASH_ATTN` out of four valid backends
  (`m-gate.log:57`, `cuda.py:486`). So the denominator ran vLLM's sixteen
  full-attention target layers on a backend vLLM itself ranks below its first
  choice, on a premise that no longer holds.

  **SAY THE DIRECTION PLAINLY: THE ERROR, IF IT IS ONE, IS IN OUR FAVOUR.** If
  `TRITON_ATTN` is the slower backend — which vLLM's own priority ordering
  IMPLIES rather than states, and which nothing here measures — then the
  denominator
  16.27918250335551 tok/s is too LOW and `0.8016987337853048` is too HIGH. An
  error that flatters us is the one nobody chases, so it is written down beside
  the number rather than left to be noticed. **The ratio is not withdrawn and no
  replacement is asserted.** What is asserted is that its denominator rests on a
  retracted premise, and that the exposure has a sign.

  **WHAT IS OWED, AND IT IS PENDING A LEASE THIS SESSION DOES NOT HAVE.** One run
  of vLLM against itself on this identical workload — same wheel, same host, same
  k, same prompts, same `max_num_seqs` — with `attention_backend=FLASH_ATTN`
  against `attention_backend=TRITON_ATTN`, each read back off the built engine as
  O22 requires. That measurement decides whether 0.8017 stands, is flattered, or
  is conservative. It also needs the developer to revisit the 2026-08-20
  declaration, because a wave must not substitute a denominator the developer
  set.

  **AND THE FIVE FA LAYERS INSIDE THE DENOMINATOR ARE NOW EXPLAINED, which #1685
  left open as three readings.**
  `vllm/v1/worker/gpu/spec_decode/dflash/utils.py:31-46` builds the draft's
  config with `backend=speculative_config.attention_backend`, UNCONDITIONALLY:
  the target's `attention_config.backend` is not carried through. The harness set
  the engine backend and not `speculative_config.attention_backend`, so the
  draft's backend was `None`, which is the auto-selection branch of
  `CudaPlatform.get_attn_backend_cls` (`vllm/platforms/cuda.py:429-496`), and it
  chose `FLASH_ATTN`. Those are the five `model.layers.64-68.self_attn.attn` in
  `evidence/vllm-arm.json`. **Two sibling speculators in the same wheel DO carry
  the target's backend through** — `dspark/utils.py:24-28`, whose comment names
  this exact hazard ("None re-runs backend auto-selection for the draft, which
  can pick a different attention class than the target; fall back to the
  target's"), and `gemma4/speculator.py:66-89`. So the DFlash path is the
  un-defended case rather than an upstream intent, which makes #1685's reading 1
  wrong AS INTENT — the inference is about intent, drawn from two siblings that
  defend against exactly this. Reading 2, "those layers fall back at runtime", is
  refuted on two counts: `FlashAttentionImpl.__init__` logged `Using
  FlashAttention version 2` at `flash_attn.py:906-914`, so an FA implementation
  was CONSTRUCTED for those layers, and the runtime half has no fallback to take
  — `flash_attn.py` in the staged wheel contains no `fallback` token at all, and
  `FlashAttentionImpl.forward` (line 970) raises `NotImplementedError` rather
  than degrading. Reading 3 is what the artifact supports.

  **THE RETRACTION INHERITS INTO THREE FILES, AND THIS IS THE ONLY ONE OF THEM
  THAT CAN CARRY IT.** `grep -rln FA_USABLE . --exclude-dir=.git` returns
  `.agents/benchmark-record.md`, this file, and `.agents/issue-index.md`. The
  index quotes `FA_USABLE=0` as a live constraint in the #1456, #1658 and #1685
  rows, and it is append-only by rule and by
  `scripts/check-issue-index-append-only.py`.
  `.agents/benchmark-record.md:3` self-declares "Append-only forensic record",
  and it carries `FA_USABLE=0` un-annotated inside the live "#1685 (new, OPEN)"
  item of the 2026-08-22 entry; note that its append-only status is a convention
  rather than a gate, which is the whole subject of
  [#1373](https://github.com/mudler/vllm.cpp/issues/1373). So this entry is
  where a reader lands instead, and the two append-only sites stay as written.
  **That `benchmark-record.md` item is STALE rather than wrong** — it says
  `FA-CONSTRAINT.txt` RECORDS `FA_USABLE=0`, which is still true, and closes
  "Unresolved.", which has stopped being true — so nothing there needs
  retracting and the retraction rides the next `SPEC-DFLASH2` entry APPENDED to
  that file. Annotating it in place would take a lock on the one file whose own
  issue says every appending pull request conflicts, to add a forward pointer.

  A grep of `.agents/oracles/` and `.agents/upstream-sync.md` for `FA_USABLE`,
  `FLASH_ATTN` and `TRITON_ATTN` exits 1 with no output, so no oracle file needs
  retracting; the plan in #1456's body to write the constraint into
  `.agents/oracles/vllm.md` was never carried out.

  **WHAT IS STILL NOT PROVEN IS THE KERNEL LAUNCH.** Construction is proven and
  coherent output is measured on both arms. A trace showing an FA2 kernel enter
  the SM on this box is not in hand and needs a lease. It is not needed for the
  retraction, and it IS needed before anybody claims the forward JIT is free.

## Now

**W6 TOOK THE GATES on 2026-08-21, on `dgx:gpu0` through an `rc` lease, and G2
and G3 both READ.** This is the first time either engine has been asked what the
OTHER's DFlash2 draft proposed.

`tests/parity/test_qwen38_dflash2_spec_decode` on device: **3 cases / 142
assertions / 0 failed / `Status: SUCCESS!` / exit 0**, against the committed
golden `tests/parity/goldens/dflash2_27b/dflash2_27b_spec_on.json`, sha256
`b051b4143d1b214d415951513c43f7148b3026a9e975f8c08d8cd3a60af7ee59` — the exact
bytes the run read.

**A RERUN TODAY WILL NOT PRINT 142, and the figure above is W6's run rather than
the current one.** The repair wave added a fourth case and its own assertions, so
the suite now reads 4 cases / **70** assertions on a box with no checkpoint (the
e2e case SKIPs) against 3 / 41 before it — 4 / 65 after the first repair wave and
4 / 70 after the second, which added the `verified` boundary case's five
assertions. The device count moves by the same delta plus the two new
ours-vs-theirs assertions. The 142 is retained because it is what was measured;
it is not a number to reproduce.

Reproduce with

```sh
VLLM_DFLASH2_TARGET=<Qwen3.8-27B dir> VLLM_DFLASH2_DRAFT=<z-lab/Qwen3.8-27B-DFlash2 dir> \
  ./build/tests/test_qwen38_dflash2_spec_decode
```

**THE TREE THAT PRODUCED THAT 142/0 IS NOT THE TREE THE RECORD PINNED, and the
repair wave identified it.** `.agents/benchmark-record.md` pins the ours side as
tree `81b530cff097db493e44e4de9a1c727530ed4467`. That tree LACKS
`Reconstructed::verified`, the per-prompt `CHECK(our_recon_here == their_acc)`
and three instrument assertions -- exactly **+8** executed assertions on this
workload. The record's own text says the pre-fix run was 134 assertions with one
failure, and `134 + 8 = 142`, so `81b530cff` is the FAILING run's tree.

The passing tree is identified rather than guessed, because the two recorded
trees differ in exactly ONE compiled file. `git diff --name-only 81b530cff
bb416e0ae -- src include tests examples tools CMakeLists.txt` returns the two
goldens (data, not compiled) and
`tests/parity/test_qwen38_dflash2_spec_decode.cpp`. So the passing binary's
compiled sources are `81b530cff`'s with that one blob replaced, and that tree is
**`0ac277b3a66b5deabe4871959f0f03566c08deda`**, reconstructible in two commands:

```sh
GIT_INDEX_FILE=$(mktemp) git read-tree 81b530cff097db493e44e4de9a1c727530ed4467
GIT_INDEX_FILE=$SAME git update-index --cacheinfo \
  100644,47c53d17a58584987599028519148747b3f018e9,tests/parity/test_qwen38_dflash2_spec_decode.cpp
GIT_INDEX_FILE=$SAME git write-tree   # -> 0ac277b3a66b5deabe4871959f0f03566c08deda
```

**What is still not known, and is stated rather than papered over:** no
`git write-tree` was taken after the fix, so no tree object recorded at the time
names the passing run. `0ac277b3` is a RECONSTRUCTION resting on the inference
above, not a value read off the run. An earlier revision of this paragraph added
that "the dispatched mutation counts W6 recorded (5 and 37)" carry the same
caveat; **that claim is DELETED rather than qualified, because it contradicts
`### W6's MUTATION SET` further down this same `## Now` section and nothing in
the tree supports it.** That subsection says W6 recorded no mutations anywhere,
`git grep "5 and 37"
bb416e0ae` returns nothing, W6's commit body carries no mutation prose, and
`git show bb416e0ae:.agents/benchmark-record.md` has no mutation table in the W6
entry. So the two counts are unfindable and the standing statement is the other
one: **W6 recorded no mutation count anywhere**, and every count in this file was
taken by a repair wave. `81b530cff` also does not contain the goldens at all,
which is CONSISTENT with the run reading one through `VLLM_DFLASH2_GOLDEN` from
the lease -- an inference, not a reading, because an untracked golden sitting in
the run's worktree fits the same evidence and no log survives to separate them.
Either way the golden's sha256 above is what pins the DATA, and it matches the
committed file byte for byte.

The FLASH_ATTN arm of the same capture is committed beside it as
`dflash2_27b_spec_on_flash_attn.json` and is selected with
`VLLM_DFLASH2_EXPECT_BACKEND=FLASH_ATTN`; it carries no per-block drafts on any
of its four records, because it was taken before the hook reached the replayed
graph.

**AND SELECTING IT USED TO PRODUCE A VERDICT ABOUT THIS ENGINE.** Through W6 the
gate's only liveness check on the golden was
`CHECK(golden.value("draft_hook_installed", false))`, which that golden passes:
the flag is `true` and every `blocks` list is empty. The run then found no block
to pair and reported "STRUCTURAL: not a single block pair was anchor-aligned
inside a shared prefix", plus a 49-vs-0 acceptance mismatch on three prompts and
a 209-against-0 total -- five failures naming our drafts, on a capture where the
oracle's instrument was what was missing.

Repaired by the W6 repair wave: `InspectGoldenDrafts` decides liveness from the
golden's own records, per record and not by total, and a drafts-less golden makes
G2b and G3 **VOID** with a message that says whose instrument is absent. G2a
(output identity) still runs, because such a golden can still answer it. The rule
is gated on every box by
`dflash2 gate: a golden's drafts decide whether G2b and G3 are takeable`, which
reads both committed goldens with no checkpoint and no GPU, and holds the flag
against the drafts in the tree as it stands today.

**THE FLASH_ATTN LABEL IS A POST-HOC RELABEL, not a read-back**
([#1562](https://github.com/mudler/vllm.cpp/issues/1562)). That golden carries
`attention_backend_source: "corrected from the run log by w6-relabel.py; the
capture's original value came from VLLM_ATTENTION_BACKEND, which does not exist
in this wheel and selected nothing"`. `## Owed` O22 lays down the rule that the
resolved backend must be read back off the built engine, and this label does not
meet it. Neither the log nor the script is committed, so nobody can re-derive it.

### G2 — SATISFIED, and STRICTLY, which the envelope did not require

| prompt | tokens exact | shared prefix | draft blocks identical |
|---|---|---:|---:|
| `The capital of France is` | YES | 64/64 | 14/15 |
| `def fibonacci(n):` | YES | 64/64 | 10/10 |
| `Q: What is 17 * 23?\nA:` | YES | 64/64 | 10/10 |
| `The three laws of robotics are` | YES | 64/64 | 11/12 |

**4 of 4 prompts token-exact** against the beyond-pin oracle, and **45 of 47
draft blocks byte-identical**. The ratified near-tie envelope permits a
divergence and none occurred on the output; it is claimed as measured rather
than as the strict form, because the DFlash1 precedent and the backend control
below both say this margin is not guaranteed.

The two draft blocks that differ each differ by ONE token, in the middle of the
proposal. `[14227 369 14227 13 198 760 6511]` against
`[14227 369 24844 13 198 760 6511]` at slot 2, and
`[39262 279 9861 2574 314 539 279]` against `[39262 279 10895 2574 314 539 279]`
at slot 2. Both blocks then emitted the same target tokens.

**THE ATTRIBUTION IS WITHDRAWN**
([#1564](https://github.com/mudler/vllm.cpp/issues/1564)). W6 wrote that both
flips are "the shape `## Port map` predicts: the lattice op is a REDUCTION over
`selector_rank`". Nothing in this capture measured that. The golden records
`{call, req_row, anchor, drafts}` per block and nothing else -- no candidate
values, no logits, no top-2 gap -- so it cannot say whether either flip was a
near-tie at all, let alone which reduction it happened in.

**And the block SHAPE argues against the op that was named.** In both blocks
only slot 2 changes and slots 3-6 are byte-identical. Per
`src/vt/cpu/cpu_ops.cpp:3219` -- "Step l reads block row `previous`, the slot
step l-1 chose" -- a flipped CHILD INDEX in `SelectorEdgeScores` moves the
predecessor row that every later step reads, so four identical later slots would
be four coincidences per block, twice. A different candidate ID at the SAME
winning slot, which is a rank swap in `ComputeCandidates`' top-k over the target
head's logits, produces exactly this shape with none. Neither is measured, so
neither is claimed here.

`SPEC-DFLASH` D6 licenses a near-tie envelope. It does not license labelling an
unmeasured flip as one, and it never licensed naming the op.

**The instrument is named and the measurement is OWED.**
`Qwen3DFlash2Model::ComputeCandidates` already returns `(ids, values)`, and so
does upstream's `compute_candidates`, so the next capture can record the top-2
candidate margin at the flipping slot on both sides. That turns the envelope
into a measurement and lets the data name the op instead of the prose.

### G3 — SATISFIED, SAME-TRAJECTORY, and IDENTICAL — but it is a COROLLARY of G2 on this capture, not a second measurement

**Read this heading before the table.** The per-prompt equality below is real,
and it is very nearly ENTAILED by G2's own result rather than independent of it.
Established by the W6 repair wave on 2026-08-21, by recomputing the oracle side
from the committed golden:

- Both engines are free-running and both emitted the SAME output on all four
  prompts, and the prompt ids are asserted equal first, so the reconstruction
  runs on the same `out` on both sides.
- 45 of 47 blocks are BYTE-IDENTICAL, so for those 45 `ReconstructAcceptance`
  runs on identical inputs and must return identical counts.
- **Both divergent blocks have their divergent slot rejected on BOTH sides.**
  Record 0 block 13: the output's slot 2 is `31785`, which is neither draft's
  candidate (`14227` against `24844`), so both accept 2. Record 3 block 6: both
  drafts carry `279` at slot 1 against an output of `9861`, so both accept 1.

Per-prompt equality therefore follows arithmetically, and so do the 216-vs-216
counter pair and the 7-token truncation deficit. The result is NOT deleted and
NOT weakened -- it is a genuine consistency check on the reconstruction, and had
either divergent block's flipped slot been ACCEPTED on one side the two counts
would have parted -- but it must not be quoted as a second, independent gate
reading. An acceptance gate that could disagree with G2 needs a prompt where the
two engines' drafts differ AND the difference reaches the accepted prefix, and
this capture contains no such prompt.

Same-trajectory by construction: all four prompts produced the SAME token stream
on both engines, so there is no trajectory confound to argue about. "By
construction" here means by ADMISSION, not by teacher forcing -- see
`## Gates` G3.

| prompt | our accepted | oracle accepted |
|---|---:|---:|
| 0 | 49 | 49 |
| 1 | 54 | 54 |
| 2 | 54 | 54 |
| 3 | 52 | 52 |
| **total** | **209** | **209** |

Both counted the same way, from drafts and output. On the OTHER instrument the
two also agree: our runner's cumulative counter reads **216** and vLLM's
`spec_decode_num_accepted_tokens` reads **216**. And the block counts agree
exactly too -- our reconstruction verifies 47 blocks, the oracle's verifies 47,
and vLLM's `spec_decode_num_drafts` reads 47.

**W6 PRINTED BOTH OF THOSE AND ASSERTED NEITHER, and the wording above claimed
them as ours-vs-theirs results.** `our_acc_sum` was a `MESSAGE` and was never
CHECKed against vLLM's counter; our verified block count was not computed at
all, and the "47" the gate printed was `draft_blocks_compared`, which is bounded
by the shared-prefix cut and is a count of blocks the gate could PAIR rather
than blocks this engine verified. What W6 actually asserted was the oracle's
reconstruction against vLLM's two counters, and the per-prompt
reconstruction-vs-reconstruction equality.

Both are asserted now, by the repair wave, guarded on `same_traj == total`
because vLLM's counters are pooled over the whole capture: `our_blocks_all ==
oracle_blocks_all` (which, chained with the already-asserted `oracle_blocks_all
== counted_drafts`, makes "our 47 and vLLM's 47" an asserted identity) and
`our_acc_sum == counted_acc`. **Neither new assertion has been executed on the
device**, because this repair wave took no lease and the case is dgx-only; they
encode numbers W6 measured and printed, and the next device run is what confirms
them.

The 209/216 gap is `max_tokens` truncating the last block of three of the four
requests, it appears on BOTH engines, and it is why the comparison is stated as
two matched pairs rather than one mixed one. The first gate run mixed them --
our counter against the oracle's reconstruction -- which is the D8 shape in
miniature, and it is corrected here.

### G4 — SATISFIED end to end, which discharges `## Owed` O17

The published `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` was loaded through
`examples/vllm-cli` against the real 27B target, proposed 7 speculative blocks
and generated. Peak RSS **44.85 GiB**, against **44.54 GiB** for the safetensors
drafter -- 0.70% apart on files that differ by 2.52 GiB, which is O13's claim
measured rather than argued. The two arms propose DIFFERENT drafts and emit the
SAME tokens, which is the acceptance-only regime this row exists to gate.

### The device arms, verified on hardware

All seven DFlash2 suites green on `sm_121a` with **zero** `no CUDA backend;
skipping` lines: `test_ops_dflash2_grouped_conv` 9936 assertions,
`test_ops_dflash2_selector_edges` 3859, `test_ops_topk_values_indices` 560,
`test_ops_dflash2_path_walk` 83, `test_qwen3_dflash2_draft` 277,
`test_dflash2_runner_reach` 86, `test_dflash2_argmax_guard` 30.

### THREE INSTRUMENT FAILURES, EACH OF WHICH READ AS A VERDICT ABOUT THE CODE

vLLM exposes no per-block draft-token counter, so G2's distinctive half needed
its speculator instrumented, and getting a hook that could SEE the drafts cost
three full 51.75 GiB loads. Recorded in `## Owed` O23 with the exact failure
text, and summarised here because the pattern is the point:

1. The engine core runs in a SEPARATE PROCESS by default, so an in-process
   monkeypatch never reaches the object that drafts. Reads as "the draft is
   empty" on a run whose counters say it drafted 350 tokens.
2. `capture_model()` calls the hooked method inside a CUDA graph capture, where a
   host copy is illegal. Reads as "the oracle cannot capture graphs with a
   DFlash2 speculator", which is FALSE.
3. And `_generate_draft` is the WRONG SEAM under the production configuration.
   `DFlashSpeculator.propose` branches on `cg_mode == FULL` and REPLAYS the
   captured graph, so the Python method is never entered. This one is the
   instructive one: the run asserted `ENGINE_CORE_CLIENT=InprocClient`,
   `HOOK_ON_CLASS=traced` AND `RESOLVED_ATTENTION_BACKEND=TRITON_ATTN`, every
   precondition it checked was TRUE, and the instrument was still blind.

Only the ABORT ON ZERO BLOCKS caught all three. The general rule, which
`.agents/verification.md` already states and this wave paid for three times: a
precondition check bounds only the failure it names, and a passing one is not
evidence that the instrument SAW anything.

**A FOURTH defect was in OUR gate rather than in the oracle hook, and the first
gate run found it.** The gate compared the oracle's RAW propose count against
vLLM's `spec_decode_num_drafts` -- 55 against 47 -- when the quantity vLLM counts
is blocks that STARTED inside the output. That single `CHECK` is the only one of
134 assertions that failed. It is corrected to the verified count, at which point
it is EXACT rather than banded, and our own trace independently reads 47.

### SPEED — TAKEN on 2026-08-22, and it is a RECORDED RATIO rather than a pass

**0.8016987337853048** ours/vLLM on `output_throughput_tok_s`, 13.051 against
16.27918250335551 tok/s, verdict **`RECORDED, no floor declared`**. `dgx:gpu0`,
`rc` job `ec9cf6cd-0aaf-4323-806d-6a12da2bd08f` (09:44Z-11:12Z), gate at
`origin/main` `d25730fbbc2afeafb9096d150823c2a4334d0619`, `GATE_RC=0`. Evidence
`/mnt/nas_share/rc/dflash2-1673/out-n1673b/`, full entry in
[`.agents/benchmark-record.md`](../benchmark-record.md) with the command that
derives each figure.

**NO BAR WAS DECLARED FOR THIS AXIS, so this is not a pass and must never be
written as one.** `floor` is `null` in the artifact and the verdict string says
what that means. The number has a reproducible provenance and no accept/reject
meaning. **THE OTHER THREE AXES ARE `NOT MEASURED`** — `ttft_ms` and
`peak_device_bytes` because `examples/cli` reports neither, `tpot_ms`
deliberately, for the reason `## Owed` O26 residual 2 gives.

Both medians fold the SAME sixteen warm legs of twenty through
`dflash2_speed_harness.fold_legs`, every leg on both arms returned its 64
completion tokens with `finish_reason: length`, and the workload fingerprint is
identical: `173f9e98...`, 4 prompts x repeat 5, k=7, `max_tokens 64`,
`max_num_seqs 1`, `temperature 0.0`, and `enforce_eager: false` on the
denominator. The oracle is the beyond-pin wheel
`0.1.dev1+g66e5414c6` (sha256 `fbc247ab...`) at `66e5414c...`; our binary is
sha256 `07b0bae6...` with the O32 leg marker found in it by `strings`. The
backend scalar `TRITON_ATTN` came from `read_back_from_engine`, not from a
relabel. Clocks: ours 86 retained samples over 16 spans at 1.0285% spread with 0
idle excluded, the oracle 83 busy with 2 idle excluded at 0.7952% spread, one
`boot_id`, `same_boot: true`, `reasons: []`, offsets +0.5169% median and
+0.2808% mean.

**THE TWO SENTENCES THIS SECTION USED TO CARRY ARE STILL TRUE AND STILL BOUND
THE FIGURE.** Our DFlash2 draft runs OFF the paged CUDA-graph fast path, because
the selector needs the hidden states of the same forward its logits came from,
while the oracle GRAPHS its draft step. What is no longer true is the second
bound: this run did NOT pay a CIFS load inside the number. `vllm-cli` marks its
own leg boundaries since O32, the folded legs are 3.744 s to 10.385 s each, and the
four ~200-290 s cold legs are discarded by name. The loads sat outside every
span.

**FIVE CAVEATS TRAVEL WITH THE RATIO, and they are the reason it is recorded
rather than claimed.**

1. **[#1673](https://github.com/mudler/vllm.cpp/issues/1673) FIRED AND DID NOT
   REFUSE, WHICH IS A PROPERTY OF THIS RUN AND NOT OF THE RULE.** Our arm's
   record covers 16 warm leg spans (`our-arm.json` `clock.window`:
   `spans 16`, `spanned_s 92.616`, `retained_samples 86` of
   `stream_samples 2943`). The oracle's record has no `window` key at all: it is
   its whole sampled stream, 85 samples over 92.51 s, covering the entire
   20-leg prompt loop with the cold legs inside it. `compare_clock_records`
   gated the median and mean offsets between those two different populations at
   1.0% each and they landed at 0.517% and 0.281%. Nothing measured that margin
   in advance and nothing guarantees the next run. STILL OPEN.
2. **[#1685](https://github.com/mudler/vllm.cpp/issues/1685), NEW: the
   denominator may not be purely what the scalar says.** Five draft layers,
   `model.layers.64-68.self_attn.attn`, resolved `FlashAttentionBackend` while
   the declared and recorded scalar is `TRITON_ATTN`, and `FA-CONSTRAINT.txt`
   records `FA_USABLE=0` for sm_12x from #1456. The engine loaded and generated
   anyway. Third independent observation of the same contradiction on this box.
   UNRESOLVED.
3. **[#1667](https://github.com/mudler/vllm.cpp/issues/1667) is untouched.**
   Both arms judge legs before the arm record reaches disk. No leg was short
   here, so nothing was lost — the defect is unchanged and the next short leg
   still costs a lease.
4. **One measured axis is not a speed gate.** Memory, TTFT and per-token latency
   remain open gaps on this row.
5. **[#1456](https://github.com/mudler/vllm.cpp/issues/1456) IS RETRACTED, so
   caveat 2 grew a SIGN: the denominator may be flattering us.** `TRITON_ATTN`
   was declared for this box because #1456 concluded the wheel's flash-attention
   cannot reach sm_12x. Both staged wheels carry `.target sm_80` PTX beside the
   SASS, and vLLM's own `supports_compute_capability` admits `>= 8.0`, so the
   premise is gone and `FLASH_ATTN` is what vLLM's production configuration
   selects here. If it is also the faster one, then 16.279 tok/s is too low and
   0.8017 is too high. Nothing is re-measured, the ratio is not withdrawn, and
   [#1796](https://github.com/mudler/vllm.cpp/issues/1796) with `## Owed` O33
   carries what it would take to settle it.

**AND O32 IS WHAT MADE THIS RUN EMIT A NUMBER AT ALL.** The whole-window summary
the sampler wrote, `evidence/clock-ours.json`, reads 550 busy of 2943 samples —
**18.688% busy**, spread 5.392% — which is below `MIN_BUSY_FRACTION`. On the
pre-O32 path this lease would have been refused exactly as `9ee9f53a` was at
18.37%. Quote the SPANNED record inside `dflash2-speed.json`, never
`clock-ours.json`.

### W6's MUTATION SET, recorded here because W6 recorded none

W4 and W5 each record their mutations in this file; W6 recorded none anywhere,
and there was no `CLAIM-SPEC-DFLASH2-W6`. Both are repaired. Every count below
was taken by the W6 repair wave on the CPU dev box at the merged tree, each
mutation applied to a file whose sha256 was taken first and restored and rebuilt
after, with the match count, `git diff --stat` and the compile rc printed for
each. The suite read **4 cases / 65 assertions / 0 failed / `Status: SUCCESS!`
/ rc 0** unmutated there (the e2e case SKIPs without a checkpoint); it read 3
cases / 41 assertions before this wave, and it reads 4 cases / 70 assertions
after the SECOND repair wave below.

The three that gate what W6 landed:

| mutation | result |
|---|---|
| `ListField` drops the last id of every list | 1 case / **7 of 65** red, `Status: FAILURE!`, rc 1 |
| `len += 1 + acc` becomes `len += acc` (the bonus token forgotten) | 1 case / **7 of 65** red, `Status: FAILURE!`, rc 1 |
| the `len + j >= out.size()` truncation guard deleted | **SIGSEGV, rc 139** |

**And the third one is the reason a doctest assertion line is not a verdict.**
Under it the run prints `assertions: 64 | 64 passed | 0 failed` -- one FEWER
assertion than green, all of them passing -- while the case failed and the
process died on a signal. `Status: FAILURE!` and the exit code are what bind;
the assertions line reads like a pass and is not one.

The four that gate what this repair wave adds, all in the new always-on case:

| mutation | result |
|---|---|
| liveness decided by the TOTAL block count instead of per record | 1 case / 1 assertion red |
| every golden declared live (`g.live = true`) | 1 case / 2 assertions red |
| `with_blocks` counts every record whether or not it carries drafts | 1 case / 4 assertions red |
| the `hook_stats` residual claimed to be 0 instead of 3 | 1 case / 1 assertion red |

### The SECOND repair wave's mutation set (2026-08-21)

W6's second fresh review returned FAIL on 2 MEDIUM and 3 LOW/INFO. Two mutations
gate what that wave landed. Both were taken on the CPU dev box at the merged
tree, on a COMMITTED tree so the restore is `git checkout --`, with the match
count, `git diff --stat` and the compile rc printed. Unmutated the suites read
`test_dflash2_runner_reach` **3 cases / 90 assertions** and
`test_qwen38_dflash2_spec_decode` **4 cases / 70 assertions**, both
`Status: SUCCESS!` / rc 0.

| mutation | result |
|---|---|
| **M1** — the startup notice reverted to its exact pre-repair `"is OPEN upstream at head 66e5414c"` wording (match 1, `+3/-5`, compile rc 0) | 1 case / **4 of 90** red, `Status: FAILURE!`, rc 1 — and they are exactly the four new assertions: `MERGED upstream` absent, `3406ec1dae…` absent, `OPEN upstream` present, `is OPEN` present |
| **M2** — `len < out.size()` becomes `len <= out.size()` on the `verified` guard | **NOT TAKEN — see below.** An OUT-OF-SUITE compilation of the extracted function shows `at_end` is the discriminating case (verified 1 under `<`, 2 under `<=`); the IN-SUITE mutation is still owed |

**M1 is the one that matters for MEDIUM-A**, because it reproduces the exact
defect: it puts the false sentence back into the binary's own startup output and
the gate now refuses it. The pre-existing 86 assertions all stayed green, so the
+4 delta is accounted for exactly and nothing else moved.

**M2 IS NOT TAKEN, and this says so rather than assuming its outcome.** It was
attempted, and the attempt was VOIDED by a harness race worth recording. Two
harness instances ran concurrently; the second took its `sha256` baseline AFTER
the first had already applied the mutation, so it found `match count: 0` and
reported `RESTORE FAILED`. The arithmetic proves the diagnosis rather than
suggesting it: the second instance's recorded "before" hash `770bee0a` is the
hash of the MUTATED file, and the clean file is `843d610b`. Neither instance
produced a usable verdict, because the first was building against a source the
second had already reverted underneath it. **The tree was never damaged** -- both
touched files were verified byte-identical to their `HEAD` blobs afterwards. The
harness now takes a `flock` before it mutates; a mutation harness with no mutual
exclusion is the same defect class as two GPU mutexes that do not exclude each
other. The
fixture it would exercise IS committed: `at_end` in
`dflash2 gate instrument: acceptance reconstructs from drafts and output` puts a
block at EXACTLY `out.size()`, which the pre-existing `tail` case never reached —
its unconsumed blocks start at len 9 against an 8-token output, so `<` and `<=`
answer identically there and the boundary was ungated. The fresh review measured
that directly: it mutated `<` to `<=` and the whole suite stayed green. What M2
would decide is whether the new case closes that hole, and the expected reading
is `edge.verified == 1` red under `<=`.

**AN OUT-OF-SUITE COMPILATION SAYS IT DOES, and the IN-SUITE mutation is still
owed.** `ReconstructAcceptance` was copied VERBATIM out of the test into a
standalone translation unit and compiled both ways on 2026-08-21
(`g++ -std=c++17 -O0`, rc 0 each). `main`, `tail` and `shift` read identically
under `<` and `<=`; `at_end` reads verified **1** under `<` and **2** under
`<=`, and a `diff` of the two runs is that one line. So the fixture DOES
discriminate and `CHECK(edge.verified == 1)` reds under M2. `## Owed` O24 carries
the table. **That is not the mutation and does not discharge it**: it measures a
copy of the function under a compiler, not the built test binary, doctest's
reporting, or whether the suite reaches the case at all -- which is what a
mutation is run for. **The in-suite mutation has still not been observed here.**

The box was starved, and a clean retake was not affordable: every mutation cycle in
this build tree rebuilds the WHOLE library (464 objects, not the one translation
unit that changed), and at loadavg 145 with 12 objects per 10 minutes measured,
one cycle is hours rather than minutes. Running the suite against the library
still on disk was REFUSED, because it had been compiled from M1's mutated source
and would have been a verdict about a tree that no longer existed. Owed:
take M2 on a quiet box. Until then the `at_end` case is committed and unproven,
which is a weaker claim than the rest of this section and is written as one.

**NOT MUTATED, and named rather than left to be assumed:** the e2e case's own
`gd.live` guards on the G2b/G3 hard bars and its two new ours-vs-theirs
assertions. That case is dgx-only and SKIPs without a checkpoint, so no mutation
of it can be executed on this box. Deleting the production call site is not
available either, for the same reason. The always-on case exists precisely so the
liveness RULE is gated somewhere it runs; the WIRING of that rule into the e2e
case is owed a device run.

---

`SPEC-DFLASH2` is `ACTIVE`. W1 landed on 2026-08-19 (the route and D4's
`is_causal` precedence). W2 landed on 2026-08-19 (the grouped dynamic depthwise
convolution, REACHED). W3 landed on 2026-08-20 (the candidate selector, reached
from the runner). W4 landed on 2026-08-20 (the path walk; a safetensors DFlash2
draft DRAFTS). **W5 landed on 2026-08-20: THE GGUF DRAFTER ARM. Both containers
draft, and the startup refusal is gone from both.**

**What W4 ships is the end of the mechanism.** `vt::Dflash2PathWalk`
(`OpId::kDflash2PathWalk`, kernel-matrix row `KERNEL-DFLASH2-PATH-WALK`) turns
the selector's lattice into k tokens: start at the verified anchor — already
present as every predecessor slot of step 0, so the walk enters at slot 0 — take
the best child, then read the NEXT step's block at the predecessor row just
chosen. It runs in UPSTREAM'S OWN GRID, one program per request with the step
loop INSIDE it, which is what `## Risks/decisions` D3 requires from the first
landing: the identical sequential walk shipped host-side in DSpark and measured
28% of the 27B draft step ([#436](https://github.com/mudler/vllm.cpp/issues/436))
before `SampleSequentialDevice` moved it. Unlike the lattice op it is specified
BIT-EXACT across backends, because it performs no arithmetic at all — only
comparisons and one gather — so there is no reduction order for a backend to
differ in.

**Two contract points decide tokens, and both are pinned by literal cases.** A
tie resolves to the LOWEST slot, which is `tl.max(..., return_indices=True)`'s
own rule and is not cosmetic: it picks the next step's PREDECESSOR row too, so it
moves the whole remaining path rather than one token. An all `-inf` row —
upstream's fully masked lane — resolves to slot 0 rather than to "no index",
which is the answer the natural parallel formulation does NOT give unless the
seed is named. Both arms reach both answers the same way: seed at `-inf` with
slot index `top_k`, keep only what STRICTLY exceeds the running best (so a NaN
never wins on either side), then collapse a `top_k` seed to 0. The CUDA arm did
not do that until W4's fresh review — its per-lane scan carried an equality
disjunct that let a lane holding `-inf` claim on the `-inf` seed — and `## Owed`
O11 carries the measurement and the repair.

**THE REFUSAL IS GONE, AND WHAT REPLACES IT POINTS THE OTHER WAY.** W1 refused
before any weight was read, W2 at the candidate selector, W3 at the path walk;
each time because a mechanism was missing and because falling back to the DFlash1
per-slot argmax would SUCCEED — well-formed tokens, a lossless verify, the
target's tokens still emitted, and only ACCEPTANCE falling, which no token gate
here can see. Nothing on the greedy arm is missing now, so
`RefuseDflash2PathWalk` is retired and `RefuseDflash1ArgmaxOnDflash2Block` takes
its place: it throws when a DFlash2 block reaches the argmax, and it is a no-op
on every DFlash1 propose. It lives INSIDE the DFlash1 sampler's own closure in
`propose_drafts_dflash` rather than beside the walk it defends, because a guard
adjacent to the call site it protects is deleted in the same edit that deletes
that call site.

**REACHABILITY IS MEASURED, and "it generated" is deliberately NOT the
assertion.** Through W3 the gate was the refusal's own text. Now the engine
generates, and a runner that dropped the DFlash2 arm entirely also generates —
that is the whole silent-wrong this row exists to remove. So
`tests/vllm/v1/spec_decode/test_dflash2_runner_reach.cpp` asserts two things an
argmax fallback cannot produce. It reads the drafts off the production
`VT_SPEC_TRACE` line at REAL fd 2 (a `std::cerr` rdbuf swap cannot see a
`std::fprintf(stderr, ...)`, and a capture that could not see the line it exists
to read would report "the propose did not run"). And it requires those drafts to
MOVE between two engines that differ ONLY in D9's `output_multiplier` and
`final_logit_softcapping` — scalars that touch nothing but the candidate VALUES
the selector's unary term reads, so the block forward, the convolution and the
block logits are identical between the two runs and the DFlash1 argmax would
answer the same for both. MEASURED at 2 of 8 blocks differing there and 5 of 6
over the wider sweep in the model suite, with both counts logged rather than
assumed.

**The mutation set that DOES redden**, each applied with its match count printed,
each with `compile_rc` printed, each restored byte-for-byte and verified by
sha256, each followed by a rebuild: dropping the walk's `previous` CARRY (op
suite 2 cases / 17 assertions red); `>` to `>=` in the argmax, i.e. the tie-break
(2/3); the all `-inf` collapse answering `K-1` instead of 0 (1/1); the gather
ignoring the winning slot (4/26); the walk's per-request row indexing (guard
suite 1/2); DELETING the runner's walk call site (reach suite 2 cases / 5
assertions red, `24` assertions run against `83` clean because the cases abort on
the throw); REPLACING the runner's walk with the DFlash1 argmax (reach 1/1 — the
D9-scalars case, and this is the GUARD-INDEPENDENT reachability proof, because it
reddens without the guard firing at all); deleting `DflashProposeBlock`'s walk
call (draft suite 1 case red); restoring the checkpoint-derived `conv_block_size`
seed (draft suite 1 case / 2 assertions red); breaking the startup notice (reach
1/1 and routing 1/1); and dropping the `final_out` capture the selector needs
(reach 2/5).

**Two mutations came back GREEN and are recorded as such rather than as passes.**
Deleting `LoadDflashDraft`'s `conv_block_size = draft->k + 1` leaves every suite
green, because the only harness that reaches a real engine enters through the
in-memory `DflashDraft` overload and bypasses that function by construction —
`## Owed` O5, where W4 also records that the line's failure mode is now a LOUD
refusal rather than a wrong answer, because `LoadQwen3DFlash` no longer seeds the
field. And deleting the argmax guard's own call leaves every suite green, which
is what a guard reachable only under another mutation looks like; its throwing
arm is gated directly in `test_dflash2_argmax_guard.cpp` and its production value
is that it converts a deleted call site into a named failure.

**What is NOT here, and it is upstream's own polarity at the moved head.** The
probabilistic draft-sample arm and its realized-q cache are not ported.
`## Risks/decisions` D13 and `## Owed` O12 carry it: `_generate_draft` calls
`_cache_draft_logits` only `if self.draft_logits is not None`, which is set only
for `draft_sample_method == "probabilistic"`, and
`vllm::ParseSpeculativeConfigJson` refuses that value BY NAME here against a
verify that is accept-iff-equal. Landing it would land two mechanisms no
production entry point can reach. The half of `draft_logits_spec` this engine CAN
honour is honoured — `None` for greedy, so no proposal distribution is allocated
and no realized score is cached on the arm that ships — and that is the reverse
of `19c93519`, where DFlash2 forced the allocation unconditionally.

**And the T>0 walk was never inverse CDF.** `## Upstream chain` said so from the
row's opening brief until this wave; at both heads the non-greedy branch draws
GUMBEL-MAX noise keyed by the candidate token ids. It is corrected in place
rather than annotated, because a wave scoped to write an inverse-CDF walk would
have written the wrong thing. `## Upstream chain`'s six-item enumeration of the
speculator's head move was RE-READ against the two blobs and is correct; the one
correction is a SEVENTH item in the base class (`draft_logits_spec`) and the
`+24 / −6` file it lives in, which the table did not list.

**W4'S FRESH REVIEW RETURNED FAIL, AND THE REPAIR IS IN THE SAME WAVE.** One
MEDIUM, three LOW and one INFO, all repaired here rather than deferred, because
each is a gate that could not say how.

*The MEDIUM was a prose guarantee with nothing behind it.* `src/vt/ops.cpp`
carries a written note above the walk's `[B,L,K,K]` check saying that BOTH
trailing axes are checked because they are the predecessor axis and the child
axis and inferring one from the other would admit a lattice indexed the wrong
way round. The reviewer proved that unmeasured: dropping either conjunct, and
deleting the whole `VT_CHECK`, each compiled clean and left all four suites
green. **THE MECHANISM RECORDED HERE WAS BACKWARDS, and
[#1518](https://github.com/mudler/vllm.cpp/issues/1518) corrects it.** This
paragraph used to say the case built a `{B,L,K,K}` tensor with `Contig(...)`,
wrote `bad.shape[2] = K - 1`, and that the resulting stride desynchronisation
made `contiguous tensors required` throw FIRST, so a bare `CHECK_THROWS` was
satisfied by a guard it was not written for. Measured on untouched production
code, that is false. The strides do desynchronise — `probe IsContiguous=0` for
`shape[2]` and `shape[3]` alike — but the walk checks shape at
`src/vt/ops.cpp:3331` and contiguity at `:3339`, in that order, so the SHAPE
guard still answered: `probe threw: vt: dflash2-path-walk: scores must be
[B,L,K,K] matching candidate_ids at src/vt/ops.cpp:3331`. The pre-repair case
DID reach the guard it named.

The true mechanism is the reverse, and it is the one worth carrying forward:
DELETING the shape guard hands the throw to `contiguous tensors required`
(measured at `src/vt/ops.cpp:3336` with the check removed), because the mutated
tensor is already non-contiguous. A bare `CHECK_THROWS` therefore still passes
with the guard gone, and the deletion is INVISIBLE — the pre-repair case form
with the whole `VT_CHECK` deleted ran **7 cases / 47 assertions,
`Status: SUCCESS!`, rc 0**. The generic rule is about what a DELETION exposes,
not about which guard fires today: build a negative input that no OTHER guard
can refuse, and match on the message. The conclusion W4 reached was right and the
repair is correct — the guard was ungated and is gated now — but an implementer
who took the old explanation as guidance would build the next negative case
against the wrong hazard. Both axes are now driven by GENUINELY CONTIGUOUS wrong-extent
lattices (`{B,L,K,K-1}` and `{B,L,K-1,K}`, built with matching strides, over a
buffer sized for the full lattice so a deleted guard reads in bounds and simply
fails to throw) and matched with `CHECK_THROWS_WITH_AS(..., doctest::Contains(
"scores must be [B,L,K,K]"), std::runtime_error)`. All three mutations now
redden: child conjunct 1 case / 1 assertion, predecessor conjunct 1/1, whole
check 1/2.

*The reviewer also asked whether the sibling shared the weakness. It did.*
`tests/vt/test_ops_dflash2_selector_edges.cpp` carried two bare `CHECK_THROWS`
that would have been answered by a neighbouring guard ONCE THE NAMED ONE WAS
DELETED — the same correction [#1518](https://github.com/mudler/vllm.cpp/issues/1518)
applies above, measured here too: with the selector's own `[B,L,K,K]` check
present the case throws `scores must be [B,L,K,K]` (`src/vt/ops.cpp:3284`), and
with it deleted the throw falls through to `contiguous tensors required`
(`:3299`), so deleting the check alone left the suite green.
It is repaired in the same shape and in the same wave: every refusal matched on
its message, plus two contiguous wrong-extent output views. Deleting that check
now reddens 1 case / 2 assertions.

*The three LOW findings.* `Dflash2WalkPath`'s candidate-set check and its
i32-range refusal were both ungated — deleted together they compiled clean and
left every suite green — and are now gated by message, reddening 1 case / 2
assertions each and 1/4 together. The `tests/CMakeLists.txt` comment above the
renamed suite still described the W2 refusal and is rewritten to describe the
inverse guard the file now holds. And the CUDA lane comparator was NOT bit-exact
with the CPU arm on a NaN-bearing row, against four written claims that it was;
`## Owed` O11 carries the measurement, the one-disjunct mechanism, the repair,
and why narrowing was rejected in favour of it.

*The INFO.* `docs/FEATURES.md` carried no DFlash2 entry at all while `STATUS.md`
and `docs/SPECULATIVE-DECODING.md` both said a DFlash2 draft drafts; this is the
wave where it starts drafting, so the entry is added rather than left to W5. And
`## Gates` G2 and the `**Oracle**` paragraph name PR head `19c93519` while the
port mirrors `66e5414c` — defensible, because the greedy answer is identical at
both, but it was not STATED as a choice. It is now, with W6 named as the owner
of the reconciliation.

**Owed after W4:** O5 (item 2, the loader's conv-geometry notice; item 1 is
structurally discharged and its remaining gap is stated precisely), O9
(`input_embedding_scale` still REFUSED — W4 re-asked and the wave does not make
it reachable), O10 (the CUDA top-k's NaN ordering, owned by
[#1489](https://github.com/mudler/vllm.cpp/issues/1489)), O11 (the walk's CUDA
arm, written and never compiled here) and O12 (the probabilistic arm). O6, O7 and
O8 stay discharged.

**W5 LANDED on 2026-08-20: THE GGUF DRAFTER ARM. Both containers now draft.**

**What W5 ships is a name for what the file already contained.** The published
DFlash2 GGUF is not a different model from the safetensors draft; it is the same
81 tensors under llama.cpp's own vocabulary, and through W4 this tree had a name
for none of the eleven that make it DFlash2. `MakeDflashGgufConfig` now carries
`conv_kernel_size`, `conv_group_size`, `selector_rank` and `selector_top_k` into
`dflash_config`, and `LoadQwen3DFlashFromGguf` resolves
`blk.N.{attn,ffn}_conv_{base,proj.weight}` and
`selector_{hidden,predecessor,successor}.weight`. The KEY SPELLING IS MEASURED
rather than mirrored from HF by hope: every DFlash2 config key the published file
writes is the HF key name VERBATIM under the `dflash.` prefix, five of five, read
off the artifact on 2026-08-20. The TENSOR names are not derivable at all —
`attention_conv` becomes `attn_conv`, `mlp_conv` becomes `ffn_conv`,
`kernel_projection` becomes `conv_proj`, `candidate_selector.X_codebook` becomes
`selector_X.weight`, and the base kernel keeps no `.weight` suffix while the
projection does — so they are read and then held against all three published
arms by an asset-gated case.

**THE STARTUP REFUSAL IS GONE ON BOTH CONTAINERS, and the notice that replaces it
is now the function's whole output.** `CheckDflash2DraftArm` no longer throws for
any DFlash2 draft. It still runs, and it still prints, for three things a user
cannot read off a checkpoint: the port is BEYOND the parity pin, no throughput
number has been taken, and a GGUF drafter is dequantized to bf16 at load. That
last one is `## Owed` O13 and is the honest cost of this arm.

**THE NAME MAP IS GATED BY CONSTRUCTION, not by reading it.** The bf16 GGUF arm
is required BIT-IDENTICAL to the same draft written as safetensors under the
published HF names — 1 120 000 bf16 elements over 30 COMPARED TENSORS, from one
set of source values written into two containers. The fixture writes 36 GGUF
tensors and the loader's q/k/v and gate/up merges leave 30 comparable slots; the
"25 tensors" this section carried through W5 is the DFLASH1 fixture's count and
was wrong (#1314 F6). Both numbers are now asserted rather than quoted —
`CHECK(gm.size() == 30)` and `CHECK(compared == 1120000)`, mutation-proven by
dropping one tensor from the comparison map (2 assertions red). A name pointed at
the wrong tensor, transposed, or dropped moves those bytes; nothing in the gate
restates the mapping. Mutation-proven: swapping the two codebook names, pointing
`attention_conv.base_kernel` at the MLP sublayer, and pointing the MLP projection
at the attention one each redden 3 cases / 6 assertions.

**AND THE QUANTIZED ARMS CARRY A LOWER BOUND, because a token gate here cannot.**
This lane dequantizes by design, so the standing trap is acute rather than
absent: a k-quant tensor that never decoded would leave every token matching and
every golden passing. The gate therefore has three parts, none of them a token
comparison — **and they are NOT three equal legs**, which is how W5 first
described them and what its fresh review corrected (#1314 F3).

L1 is a PRECONDITION ON THE FIXTURE. It reads the ggml type and the byte count
off the tensor table of the file the test has just written — MEASURED at 22
quantized tensors, 1 114 112 elements in 626 688 bytes for Q4_K and 1 183 744 for
Q8_0, against 2 228 224 for the same tensors in bf16. What it proves is that the
bytes L2 goes on to compare really are block-encoded; it says nothing about the
loader, and a loader that hashed the bytes and returned garbage passes it
unchanged.

L2 IS THE BOUND. The VALUES, bit-for-bit over 266 240 elements against an
expectation the test computes from the block integers it CHOSE — the suite
carries its own Q8_0 and Q4_K encoders, including the inverse of
`get_scale_min_k4`'s 6-bit scale pack, so the comparison is a round trip through
two implementations rather than a shared helper agreeing with itself. Every
decoder mutation this row records lands here.

L3 is a cheap COROLLARY of L2 that reaches further: DIFFERENCE from the bf16 arm,
at 7 of 7 quantized DFlash2 tensors and 2048 of 2048 block logits, with the F32
tensors no arm quantizes required IDENTICAL so it cannot be satisfied by a loader
that garbled everything. The ordering is exact: a byte-hashing loader passes L1
and L3 and fails L2 hard.

**AND THE FIXTURE'S OWN COVERAGE IS COUNTED, ONE COUNTER PER PACKED FIELD**,
because L2 can only bound what the fixture drove. W5 shipped one instance of
that defect and its review found two more (#1314 F1). Measured over this
fixture's own source values, the Q4_K encoder produced a scale of EXACTLY 3 for
all 17 408 low-half sub-blocks — two bits of a six-bit field — so
`get_scale_min_k4`'s `*d = q[j] & 63` mutated to `& 15` left the suite at 9 cases
/ 4720 assertions / `SUCCESS!` / rc 0. The min field carried a weaker form of the
same thing: 36..38 in every sub-block, which catches `& 15` but not `& 47`, and
that mutation also passed. The repair drives sub-block 0 to 19 and sub-block 1 to
63 (the whole-field half), pins the min of sub-blocks 2 and 6 at 63, and
DELIBERATELY leaves sub-blocks 2 and 3 at their data-derived scale so the 4-bit
quant still spans 0..15 in BOTH nibble positions — lifting all four, the obvious
repair, collapses q to 0..2 and trades the fix for the same defect one field
down. All five counters are asserted (`scale_bits`, `min_bits` = 63 per half;
`nibble_bits` = 15 per position; plus the two above-15 counts) and all of it is
mutation-proven, both ways: `& 15` and `& 47` now redden 1 case / 7 assertions
each, removing the scale lift reddens 2, removing the min pin reddens 2, and the
naive all-four lift reddens the nibble precondition at 2.

**Two guards land with the arm, and both are gated by message rather than by a
bare throw.** All four DFlash2 geometry keys are REQUIRED once the file is
classified — `IsDflash2Gguf` answers on any one of three, so a subset would
otherwise be admitted with a guessed `conv_group_size`, which sizes the
projection wrong and is acceptance-only. And the selector's codebooks must SPAN
the target's head: W5 is the first wave where the two numbers can disagree,
because a GGUF draft declares no vocabulary and is sized from its own
`tokenizer.ggml.tokens` while the ids that index the codebooks come from the
target's `lm_head`. A mispaired checkpoint used to be an out-of-range read of a
127 MB tensor; it is now a refusal naming both widths. The vocabulary is taken
from the tokenizer and NOT from the codebooks, because a check that read its own
expectation off the tensor under test would report a transposed or truncated pair
as correct.

**D9's OUTPUT SCALARS are gated on this container too**, against declared values
rather than defaults, even though NEITHER published DFlash2 GGUF sets them — that
absence is exactly why, since a gate built from those files alone measures the
default path and reports it as coverage. `input_embedding_scale` stays REFUSED BY
NAME here as it is on the safetensors arm, with the equal-to-default arm asserted
not to refuse.

**Owed after W5:** O5 and O14 (the loader's GGUF and DFlash2 lines are still not
reachable from a production entry point, for the same structural reason, with the
bounded consequence named), O9 (`input_embedding_scale`), O10 (the CUDA top-k's
NaN ordering, [#1489](https://github.com/mudler/vllm.cpp/issues/1489)), O11 (the
walk's CUDA arm, never compiled here), O12 (the probabilistic arm), O13 (a GGUF
drafter's bf16 residency, measured at 3.584 GiB against 1.06 GiB on disk), and
three of the four the W5 REVIEW added: O15 (the three output-scalar GGUF key
spellings are inferred and no published file declares any of them), O16 (the
codebook-span guard is `==` and upstream's own condition is unread here) and O17
(no wave has LOADED a published artifact — the asset case reads headers and the
fixture runs at `H = 256`). O6, O7, O8 and O18 stay discharged; O18 was FIXED on
2026-08-21 rather than deferred, because the bit-exactness cost its deferral was
built on is not paid by a second power of two. O19 and O20 are recorded CLOSED on
arrival, not owed: the coverage counters now read the same tensor population the
comparison reads, and the asset-gated case reports each half's absence by its own
variable instead of only when both are unset.

**Owed after W6:** O5 and O14 (the loader's GGUF and DFlash2 lines are still
not reachable from a production entry point), O9 (`input_embedding_scale`), O10
(the CUDA top-k's NaN ordering,
[#1489](https://github.com/mudler/vllm.cpp/issues/1489)), O11 (the walk's CUDA arm — its own entry
already records that it has RUN on `dgx:gpu0`, and W6 ran all seven DFlash2
suites there with zero skips; what is stale is the "never compiled here" gloss in
the `Owed after W5` line above, which this line supersedes), O12 (the probabilistic arm), O13
(a GGUF drafter's bf16 residency, 3.584 GiB against 1.06 GiB on disk), O15 (the
three output-scalar GGUF key spellings), O21 (now a MERGED upstream, #1538 and
#1561), O22/O23 (HALF discharged 2026-08-21: the instrument is committed, the
run is not taken -- attempted 2026-08-22 and it measured the instrument, see
O30), O26, O27, O28, O29, O30 and O31. O16 is SETTLED by W6. O6, O7, O8, O18, O19 and O20 stay
discharged.

**And five things this row owes that W6 did not name**, all opened by the repair
wave on 2026-08-21:

1. **The gate head is one merge behind vLLM's `main`.**
   [#1561](https://github.com/mudler/vllm.cpp/issues/1561). `## Gates` G2's own
   rule selects `b389ac29`; the W6 capture is dated at `66e5414c` and re-reading
   G2/G3 at the merged head is owed. O21's parked D12 decision comes due with it.
2. **The oracle capture harness is COMMITTED as of 2026-08-21, and the
   measurement is still not taken.**
   [#1562](https://github.com/mudler/vllm.cpp/issues/1562). O23's table names
   where each described element now lives:
   `tools/bench/dflash2_speed_harness.py` (the preconditions and every refusal),
   `tools/bench/dflash2_oracle_capture.py` (the hook on `propose`, the in-process
   client assertion, the backend read-back), `tools/bench/dflash2_our_arm.py`
   (our arm, through the public ABI only), `scripts/dflash2-speed-gate.sh` (the
   lease-side procedure, marker written from `trap ... EXIT`) and
   `tests/tools/test_dflash2_speed_harness.py` (CPU, no GPU and no wheel). **No
   case count is written here any more.** It has been wrong three times -- 100,
   then 104, then 115 -- because a number about one file, stored in another,
   goes stale on every edit to the first, which is the record shape `AGENTS.md`
   §Records forbids. Derive it instead:
   `python3 -m unittest tests.tools.test_dflash2_speed_harness 2>&1 | grep '^Ran '`.
   `w6-relabel.py` and the W6 run log are recorded LOST WITH THE LEASE
   rather than owed. **The run was ATTEMPTED on 2026-08-22 and it measured the
   INSTRUMENT**: the gate as committed could emit no ratio by any path, four
   issues were filed, and O30 holds what each one stopped and what stays owed.
   So O26's run is still owed, on the head #1561 selects. That run is the only
   thing that can give the FLASH_ATTN golden's label a provenance, and the
   capture emits the golden's own shape, so its output can become that golden
   rather than merely describe one.
3. **The two divergent draft blocks are UNATTRIBUTED.**
   [#1564](https://github.com/mudler/vllm.cpp/issues/1564). The instrument is
   named — the top-2 candidate margin at the flipping slot, on both sides — and
   the next capture takes it.
4. **Three propose calls in the TRITON_ATTN golden are unaccounted for.**
   `hook_stats` says 59 calls less 1 dummy; 55 blocks are recorded. Bounded
   one-sidedly by the gate now, and unexplained, under #1562.
5. **The two new ours-vs-theirs assertions have never run on a device.** They
   encode numbers W6 measured and printed; the next device run confirms them.

Next action: **#1561** — move the gate head onto merged upstream `b389ac29` and
re-read G2 and G3 there, taking #1564's top-2 margin in the same capture and
committing #1562's harness with it — then `## Outcome`. **No throughput number
is claimed by this row and none is admissible yet**: `## Gates` defers every
ratio until G2 and G3 read at the head the rule selects, and a DFlash2 draft is
additionally off the paged CUDA-graph fast path, because the selector needs the
hidden states of the same forward its logits came from.
