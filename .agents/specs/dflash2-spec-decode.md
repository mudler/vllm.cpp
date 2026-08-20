# SPEC-DFLASH2 — DFlash2: the grouped dynamic convolution and the candidate selector

**Row:** `SPEC-DFLASH2` (engine-matrix, speculative decoding).
**Issue:** [#1314](https://github.com/mudler/vllm.cpp/issues/1314).
**Predecessor:** [dflash-spec-decode.md](dflash-spec-decode.md) (`SPEC-DFLASH`, DONE).
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

**Not merged upstream.** Read at
[vllm-project/vllm#52816](https://github.com/vllm-project/vllm/pull/52816) head
`66e5414c6d75a8529473d977f7458c140bbab8a0`, base `9842d701`, opened 2026-08-18.

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
   the verified anchor. At T>0 the walk is by inverse CDF and returns q over the
   K candidates, which the lossless verify requires.

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
| `dflash2/speculator.py` walk kernel | `src/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.cpp` + a CUDA walk kernel | D3: device from day one |
| `dflash2/speculator.py` draft-logit cache | same file | the realized q the rejection sampler reads at T>0 |
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
- G2: draft-token identity against vLLM at PR head `19c93519` on
  `z-lab/Qwen3.8-27B-DFlash2` over `Qwen/Qwen3.8-27B`, identical prompts and
  identical k, greedy. The DFlash near-tie envelope applies: `SPEC-DFLASH`
  established that strict token identity is bf16-irreducible on portable
  kernels, so the ratified near-tie gate is the admissible form and a strict
  claim is not.
- G3: **acceptance**, measured SAME-TRAJECTORY — both engines teacher-forced on
  identical tokens. `SPEC-DFLASH` D8 spent a whole campaign on an acceptance
  deficit that was a divergent-trajectory measurement confound, and D9 refuted
  it. This gate does not repeat that mistake.
- G4: the GGUF drafter arm, with the lower bound of `## Tests to port`.
- G5: reachability, as above.

**Speed.** No ratio is claimed by this row until G2 and G3 read. When one is
taken it uses vLLM's production configuration as the denominator, never
`--enforce-eager`, on an idle leased host with a same-binary A/B, per
`.agents/benchmarking.md`. Upstream's H200 numbers are not a floor for this
engine and are not treated as one.

**Oracle.** The pinned oracle cannot run this architecture. The gate oracle is
vLLM built at `19c93519`, recorded with its measured runtime version, and it is
a BEYOND-PIN oracle rather than a pin advance. If #52816 merges before the
implementation lands, the anchors move to the merge commit and this section is
reconciled rather than reinterpreted.

## Dependencies

- A GPU lease on a fleet device (`rc run`/`rc hold`) for every G2-G5 run. Never
  `ssh` to a fleet box.
- The 27B target and the DFlash2 drafter resident where the gate host can read
  them, with a recorded revision, since a repo id alone is not a pin.
- A vLLM build at `19c93519`. The parity pin stays where it is.
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
- **W4 — the speculator.** The device path walk, the realized-q draft-logit
  cache, the T>0 inverse-CDF arm, and the wiring that makes a DFlash2 draft
  reach it. G5 lands here. **Read `## Upstream chain`'s speculator table
  first**: `speculator.py` is +37 / −34 between the two PR heads W1/W2 and W3
  were written against, and every one of those six changes is math or state
  layout in W4's own territory — the `SAMPLE_PROBABILISTIC` constexpr, the
  `gumbel_noised_argmax` routing, the deleted `_selector_tokens`, the optional
  `draft_logits` with its new fp32 `draft_logits_spec`, and the conditional
  `_generate_draft` tail. W3 reconciled the head for the SELECTOR file only.
- **W5 — the GGUF drafter arm**, with its lower bound.
- **W6 — the gates.** G2 and G3 on a leased GPU against the PR-head oracle,
  then `## Outcome`.

## Risks/decisions

- **D1 — mirror the unmerged PR now, rather than waiting for it to merge.**
  Developer decision, 2026-08-19. The mechanism is architecturally separate from
  DFlash1, and two open pull requests whose second one only fixes a guard suggest
  a settled design. Cost: the anchors can move under review, and the port
  reconciles if they do. Precedent: `SPEC-DSPARK-QWEN3-ROUTING` toward
  vllm#52197. This does NOT advance the parity pin.
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
  It is not decoration here. The safetensors arm refuses a non-BF16 head one
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
  GGUF arm KEEPS the startup refusal, because its drafter arm (W5) has no conv
  weight path at all and admitting the file would load a DFlash1 draft out of a
  DFlash2 checkpoint.

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

- **O5 — `LoadDflashDraft`'s own DFlash2 lines are not gated.** Owner: this row,
  discharged by W4. Issue
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


## Now

`SPEC-DFLASH2` is `ACTIVE`. W1 landed on 2026-08-19 (the route and D4's
`is_causal` precedence). W2 landed on 2026-08-19 (the grouped dynamic depthwise
convolution, REACHED). **W3 landed on 2026-08-20: the CANDIDATE SELECTOR, reached
from the RUNNER.**

**What W3 ships is the choosing, up to but not including the walk.** Two `vt`
ops: `Dflash2SelectorEdges` scores the transition lattice
`edge(b,l,p,c) = unary[b,l,c] + <pred[pid(b,l,p)] * project(h)[b,l], succ[cand[b,l,c]]>`,
with the request's verified ANCHOR as every predecessor slot at step 0 and the
previous step's candidate at every later one; `TopKValuesIndices` is the
vocabulary top-k that EMITS (id, value) pairs, extending the sort-free
pivot-bracket threshold search this repository already carries rather than
porting FlashInfer's 3380-line radix kernel (D2). Around them,
`Qwen3DFlash2Model::ComputeCandidates` applies the org-vocab rebase,
`output_multiplier` and `final_logit_softcapping` to the candidate VALUES in
upstream's own order, and `::SelectorEdgeScores` runs the hidden projection and
the lattice. The three selector tensors load through the production loader under
the published names.

**THE TIE-BREAK IS THE CONTRACT, not an implementation detail.** The threshold
search converges to an exact array VALUE, so the survivor set keeps whole tie
groups and can hold more than k members; something has to choose among equals,
and choosing differently reorders the selector's candidate slots and moves
acceptance without raising anything. The order is descending value, ties by
ascending index — `torch.topk`'s CPU order and what FlashInfer's
`deterministic=True` exists to provide — and hand-written rows pin it, including
a tie group STRADDLING the k-th boundary, which is the one the search actually
has to resolve. A GB10 has now run those rows on the device arm too and they
agree (#1489); the ONE row of that table the two arms do not share is NaN
ordering, which the CUDA kernel cannot produce and which #1489 owns.

**The REFUSAL moved again, and the duplicate that made W2's O7 possible is
gone.** `RefuseDflash2CandidateSelector` is retired. Both propose paths —
`GPUModelRunner::propose_drafts_block` and `DflashProposeBlock` — now call ONE
`Dflash2SelectCandidates`, so the sequence a user arrives through and the
sequence a gate drives are the same implementation rather than two copies. The
boundary is `RefuseDflash2PathWalk`, and it TAKES the scored lattice: the message
names how many transitions were scored over how many requests, steps and
candidates, which tells a user how far the port got and makes the selector's
execution observable at the call site.

**O7 IS DISCHARGED, AND ITS RECORDED REASON WAS WRONG.** W2 said a gate on the
runner needed an on-disk target plus an on-disk draft through the loader plus a
populated device KV store, and that this was W4's harness. It needed a fifteen-line
overload: a way to hand a `DflashDraft` to a `LoadedEngine` built from in-memory
weights, which is the seam `mtp_weights` already had for exactly this reason.
`test_dflash2_runner_reach.cpp` now drives a real engine over a synthetic
Qwen3.5-dense target and an in-memory DFlash2 draft, generates, and reads the walk
refusal's own text: `scored-transitions=27 requests=1 steps=3 top_k=3`. Those
counts ARE the selector's output, so one string proves the runner entered
`propose_drafts_block`, that the block forward ran there, and that the selector
ran there. Two waves shipped production code nothing could reach because that
reason went unchallenged; it is corrected in `## Owed` rather than annotated.

**D9's scalars are gated against values that DIFFER from the defaults.**
`z-lab/Qwen3.8-27B-DFlash2` sets neither `output_multiplier` nor
`final_logit_softcapping`, so a gate built from that draft alone measures the
default path and reports it as coverage. The model suite drives BOTH arms and
asserts the exact formula at Muse Glimmer's 0.19611613513818404 and 20.0, in
upstream's order (multiply first, then cap the SCALED value). It also asserts the
half that makes them matter: the two scalars touch the UNARY term and not the
codebook contraction, so the arms do not differ by a monotone rescale — the
argmax over children FLIPS under some predecessor, which is what moves the path
the W4 walk takes. The flip counter carries its own precondition, because a
counter that cannot read zero measures nothing.

**The bf16 ROUNDING PLACEMENT has an executing assertion, which is the hole W2's
second review found in the convolution's evidence.** Upstream materializes two
bf16 tensors inside the einsum chain and this op rounds at those two points and
nowhere else; on f32 both roundings are the identity, so no f32 case can see the
policy. The literals in the lattice suite are chosen so the three candidate
placements answer differently: ours 7.71875, round-once-at-the-end 7.6875,
unrounded 7.699830055236816.

**Two things about the upstream head, and one about ours.** The PR head moved to
`66e5414c6d75a8529473d977f7458c140bbab8a0` (#1404); the selector's math is
byte-identical at both, and the only two changes are `set_model_tag` (a
deliberate NON-PORT, D11) and the widened LM-head guard (PORTED, D12). Ours is
that the LM-head guard is LIVE here rather than decorative: a GGUF target's
`output.weight` reaches this lane DEQUANTIZED, and the selector consumes the
target head's EXACT top-K, so `Qwen3DFlashWeights` now carries
`lm_head_dequantized` and `RefuseQuantizedDflash2LmHead` reads it.

**O8 is ANSWERED, not deferred.** The context-KV precompute applies no
convolution, and the reviewer asked whether that stays equivalent now that a
convolution exists. It does, because it is not a shortcut: upstream's
`precompute_and_store_context_kv` projects every layer's context K/V from one
shared `rms_norm(context_states, hidden_norm)` too, `DFlash2Qwen3Model` does not
override it, and `DFlash2Speculator` inherits the propose that calls it. Our
`PrecomputeContextKVDevice` is the port of that function and not a divergence
from it.

**FIVE gate weaknesses were found BY the mutation pass, not by reading, and all
five are repaired.** W2 recorded three of the same kind; this is the fourth
consecutive wave on this row where running the mutation found something reading
the test could not. Four were found by the wave's own pass and the fifth by its
fresh review, and it is counted here rather than filed elsewhere because the
count is the point: reading did not find any of them. Each came back GREEN and
had to be fixed before the wave could land:

- **The two CODEBOOKS were swappable at load.** Exchanging the
  `predecessor_codebook` and `successor_codebook` reads in `LoadQwen3DFlash`
  compiled clean and left EVERY suite in this repository green. The two share a
  shape, so the swap loads, scores every transition with the roles reversed, and
  is invisible: the verify is lossless, the emitted tokens stay the target's,
  only acceptance falls. Repaired by asserting the loaded BYTES against the
  fixture's own two generators, which is a non-circular check on which tensor
  landed where.
- **The selector-key requirement was gated by an INCIDENTAL throw.** The case
  that erases `selector_rank` used a bare `CHECK_THROWS`, and it passed with the
  guard removed — because an absent key leaves `rank` at 0 and the
  `hidden_projection` shape check then throws for an unrelated reason. A gate
  that cannot tell the refusal it means from another one measures nothing.
  Repaired by asserting the message.
- **The selector SHAPE assertions were unexercised.** Removing them left every
  suite green; a transposed codebook is a tensor that still loads and still
  produces finite scores. Repaired with a case that writes one.
- **The `lm_head_dequantized` CARRY was ungated.** Making
  `LoadGgufSharedEmbedAndHeadBf16` always report `false` left every suite green,
  so D12's guard had a live trigger that nothing proved was connected to it.
  Repaired with a two-arm case over a real GGUF fixture — bf16 head and Q8_0
  head — because a carry that always reported `true` would satisfy the DFlash2
  guard while refusing every bf16 GGUF target the DFlash1 lane has shipped.
- **And the LINK between those two ends was ungated too — the fifth, found by
  the wave's FRESH REVIEW after the four above were repaired.** Deleting the
  THIRD ARGUMENT of
  `shared.LoadInto(&…embed_tokens, &…lm_head, &…lm_head_dequantized)` in
  `LoadDflashDraft` compiled clean and left all 38 dflash/gguf suites green after
  a full `libvllm` relink, including `test_qwen3_dflash2_draft` 214/214 and
  `test_gguf_keep_quant` 6093/6093, every one `Status: SUCCESS!`. The wave gated
  the setter (which reddens) and the reader (gated directly) and not the wire
  between them, which is what makes D12's guard live. The cause was a defaulted
  parameter — `bool* head_was_quantized = nullptr` — so the deletion silently
  turned the carry off. Repaired by REMOVING the default rather than by adding a
  case: both callers now name the argument and dropping it is a compile error.
  Recorded in `## Owed` O5, which is where the two remaining ungated
  `LoadDflashDraft` lines live.

**The mutation set that DOES redden**, each restored byte-for-byte and verified
by sha256, each with its compile status printed and its application checked at
the BYTE level rather than by `git diff` (three of these files are new in this
wave and untracked, and `git diff --stat` prints nothing for an untracked file —
an applied-check that cannot see the change it checks): the lattice's bf16
rounding placement; the anchor arm; the predecessor off-by-one; the unary
broadcast axis; the top-k tie-break; the org-vocab padding mask; the org-vocab
rebase; `output_multiplier`; `final_logit_softcapping`; the ORDER of those two;
the sample-row gather; the `input_embedding_scale` refusal; the dequantized-head
guard and its carry; the selector-key requirement; the codebook shapes; the
codebook roles; the RUNNER's `Dflash2SelectCandidates` call; the runner's whole
DFlash2 arm; the runner's walk refusal; the runner's `final_out` capture; and
`DflashProposeBlock`'s own call. It does NOT redden for the TWO lines inside
`LoadDflashDraft` that no entry point this repository can drive reaches:
`conv_block_size = k + 1` and the conv-geometry notice beside it. Both are O5,
which the wave's first draft of this sentence named as one line. The third
DFlash2 line in that function — the `lm_head_dequantized` carry — was ungated
too, and is now a compile error to delete rather than a mutation the inventory
has to cover.

**The repair delta adds three mutations of its own and removes one vacuous
assertion.** Reverting the NaN comparator still reds the CPU literal case
(counts above). Forcing the device-case exclusion `RunsOnCuda` to `true` reds it,
and forcing it to `false` reds it — a filter that matched nothing would put the
NaN row back on a device that cannot answer it, and one that matched everything
would leave both device cases asserting nothing, and neither shape can fail on a
host with no `nvcc`, so both are asserted on the CPU arm. Every arm printed
`compile_rc=0`, every mutation printed its MATCH COUNT before the build, and
every source was restored `sha256`-verified. The removal is
`CHECK(flips >= 0)` in `test_qwen3_dflash2_draft.cpp`, which no `int` counter
could violate and which existed only to give a doctest `INFO` something to attach
to; the INFO is scoped to the block and stays live for the per-block
`count_flips(sp, sp) == 0` precondition below it. Putting the five copies back
moves that suite from `26 cases / 232 assertions` to `26 / 237`, all passing,
which is what shows the deletion took those five and nothing else.

**THE KERNELS HAVE NOW RUN ON A DEVICE, and one row of one table disagreed.**
The operator took a lease on `dgx:gpu0` (GB10, sm_121a, `nvcc` 13.0) and ran the
DFlash2 suites at this row's W3 head `b29b6f886`:
[#1489](https://github.com/mudler/vllm.cpp/issues/1489). `BUILD_RC=0`,
`COMPILE_ERRORS=0`, and — the precondition without which a green proves nothing
here — **zero `no CUDA backend; skipping` lines**, with
`test_ops_topk_values_indices` reporting 562 device assertions against 202 on the
CPU-only build. **O6 is DISCHARGED**: W2's convolution and both W3 ops compile,
and five of the six suites pass. **The tie divergence O10 called the real risk
did NOT materialise** — the straddling group, the group larger than k, the ties
inside the kept set and the `-inf`-saturated row all agree across the two
algorithms.

**What failed is NaN ordering, and it is repaired by NARROWING the gate rather
than by recording the divergence again.** `test_ops_topk_values_indices` read
`7 cases | 5 passed | 2 failed`, `assertions: 562 | 550 passed | 12 failed`,
`Status: FAILURE!`, every one of the twelve on the literal row
`"NaN sorts first, as torch.topk does"`, including the direct cross-arm pair
`gpu.indices[i] == cpu.indices[i]` reading `2 == 1`. The CUDA arm cannot select a
NaN at all: `TopKValuesIndicesRowKernel`'s bracket uses `fmaxf`/`fminf`, which
return the non-NaN operand, and its survivor pass tests `r[j] > thr`, which is
false for a NaN. So W3 shipped a suite that reds on any CUDA build, while
`include/vt/ops.h` asserted an ordering no shipped backend delivers — two results
for one rule, which AGENTS.md `## Gates` does not permit. The row is now excluded
BY NAME from both device cases and kept on the CPU arm where it is the
guarantee; `ops.h` states the asymmetry beside the contract; and #1489 owns
reconciling the kernel, which needs a lease and is not attempted from a host with
no `nvcc`. Nothing a user can obtain changes: no shipped path feeds this op a NaN
logit.

**Owed, and none of it is a claim wearing a pass.** O5 (`LoadDflashDraft` carries
TWO ungated DFlash2 lines, not one — `conv_block_size = k + 1` and the
conv-geometry notice — because the new harness enters through the in-memory
overload and bypasses that function by construction; its third line, the
`lm_head_dequantized` carry, was ungated as well and is now structurally
undeletable, and deleting the notice narrowed the startup discriminator from
`conv_taps > 0` to the declared architecture), O9 (`input_embedding_scale` is
REFUSED rather than implemented, because no published checkpoint can reach an
implementation and a silent omission is the defect class this row exists to
remove), and O10 (reduced to ONE measured item: the CUDA top-k does not order NaN
first, owned by #1489, with the compaction's silent `slot >= k` truncation still
named and still unexhibited). O6 is DISCHARGED.

**One user-facing message in the loader said the opposite of the other, and the
stale one named W3.** `LoadDflashDraft` still appended "the candidate selector is
NOT implemented and this draft will be refused by name at its first propose
(SPEC-DFLASH2 W3, #1314)" to its conv-geometry line — the wave that shipped the
selector, named as still owing it — while `CheckDflash2DraftArm`'s notice, in the
same file and correct, said the selector runs and the PATH WALK is W4's. A user
loading a real DFlash2 directory got both. Neither gate could see the second one,
because the harness that reads `cerr` enters through the
in-memory `LoadedEngine` overload. D10 pays for the moved refusal WITH that
notice, so this was the obligation D10 records and not a wording problem. The
boundary now has one owner: `CheckDflash2DraftArm` runs ahead of every path that
reaches `LoadDflashDraft` and its notice is gated in two suites, so the loader
line prints only the resolved conv geometry, which is the one thing it knows.

**THE TIE CASES DID NOT RUN WHERE THE TIE RISK IS.** O10 named the top-k's tie
handling as the port's real divergence risk and said the hand-written cases would
exercise it once the CUDA arm could run. They could not: every one of them called
a helper that builds a CPU queue, and the single device case ran four LCG shapes
under a comment claiming the generator repeats values at that width. It does not
— reproduced in exact float32, in Python and again in C++ which agree, those four
shapes hold 513/513, 128/128, 200/200 and 64/64 DISTINCT values per row, with the
k-th largest at multiplicity 1 everywhere, and the bulk-sort case's `{5,257,16}`
is the same. The literals now live in one `LiteralRows()` table that BOTH arms
iterate — the CUDA arm skipping the single NaN row it does not implement — and
both assert the literals rather than each other, because two implementations
agreeing on a wrong tie rule is exactly what asserting one against the other
cannot catch. #1489 then RAN that table on a GB10, which is the run this
paragraph was written in anticipation of: the tie rows agreed.

**And the CPU comparator was undefined behaviour on a NaN row.**
`if (src[a] != src[b]) return src[a] > src[b]; return a < b;` makes NaN compare
EQUIVALENT to every value while those values are not equivalent to each other —
an intransitive equivalence, which is not a strict weak ordering and is UB in
`std::partial_sort` rather than a merely surprising answer. NaN now sorts FIRST
on the CPU arm, which is `torch.topk(largest=True)`'s own order, and a table row
pins it: before the fix that row returned `{2, 1, 3}` where the contract is
`{1, 2, 3}`. **The red-before COUNTS W3 first recorded here were wrong**, and the
wave's fresh review could not reproduce them: the entry said
`199 assertions / 2 failed`, and reverting the comparator at `b29b6f886` gives
`7 cases | 6 passed | 1 failed`, `assertions: 202 | 198 passed | 4 failed`,
`Status: FAILURE!`. Both numbers were wrong; the qualitative claim was not.
RE-MEASURED in the repair delta, which adds eight CPU assertions of its own, one
build directory on the CI recipe, x86_64, `compile_rc=0` on every arm and the
source restored `sha256`-verified: with the comparator reverted,
`7 cases | 6 passed | 1 failed`, `assertions: 210 | 206 passed | 4 failed`,
`Status: FAILURE!`, all four `logged: row "NaN sorts first, as torch.topk does"`;
restored, `7 cases | 7 passed`, `assertions: 210 | 210 passed`,
`Status: SUCCESS!`. The counts are corrected rather than annotated, because a
number nobody can reproduce is what the next reader spends a build finding out.
No shipped path feeds this op a NaN logit, so the row is synthetic in the sense
the padding row is.

**D9's flip margin was 1 of 9, and it is now 8 of 45.** The half of D9 that
matters is that the argmax over children FLIPS between the two scalar arms, not
that the values differ. One synthetic block gives nine predecessor slots and
exactly one of them flipped, so `> 0` was one arithmetic accident from passing
for the wrong reason and nothing let a reader see it. The case sweeps five blocks
now, pins the slot count exactly, floors the flips under the measured values, and
keeps the zero-reading precondition per block.

Next action: W4, the speculator — the device path walk, the realized-q
draft-logit cache, and the T>0 inverse-CDF arm. It is the wave that lifts the
refusal W3 leaves behind, and it inherits a production entry point that a gate
can now reach.
