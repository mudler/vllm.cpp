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
- G2: draft-token identity against vLLM at PR head `19c93519` on
  `z-lab/Qwen3.8-27B-DFlash2` over `Qwen/Qwen3.8-27B`, identical prompts and
  identical k, greedy. The DFlash near-tie envelope applies: `SPEC-DFLASH`
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
- G3: **acceptance**, measured SAME-TRAJECTORY — both engines teacher-forced on
  identical tokens. `SPEC-DFLASH` D8 spent a whole campaign on an acceptance
  deficit that was a divergent-trajectory measurement confound, and D9 refuted
  it. This gate does not repeat that mistake.
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

**Oracle.** The pinned oracle cannot run this architecture. The gate oracle is
vLLM built at `19c93519`, recorded with its measured runtime version, and it is
a BEYOND-PIN oracle rather than a pin advance. If #52816 merges before the
implementation lands, the anchors move to the merge commit and this section is
reconciled rather than reinterpreted. The head named here is the same deliberate
choice G2 states, with the same reason and the same owner: the port mirrors
`66e5414c`, the greedy answer is identical at both heads, and W6 reconciles the
oracle build to one head when it takes the gate. No oracle has been BUILT at
either head yet, so nothing is invalidated by moving it.

## Dependencies

- A GPU lease on a fleet device (`rc run`/`rc hold`) for every G2-G5 run. Never
  `ssh` to a fleet box.
- The 27B target and the DFlash2 drafter resident where the gate host can read
  them, with a recorded revision, since a repo id alone is not a pin.
- A vLLM build at `19c93519`. The parity pin stays where it is. **That head is
  the same DELIBERATE CHOICE `## Gates` G2 and `## Oracle` state, and W6 owns
  the reconciliation**: the port mirrors `66e5414c`, which superseded
  `19c93519` on 2026-08-19
  ([#1404](https://github.com/mudler/vllm.cpp/issues/1404)), and the greedy
  answer is identical at both heads. This line named the head bare until
  [#1518](https://github.com/mudler/vllm.cpp/issues/1518), which reads as a
  leftover beside the other two.
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

- **O16 — the codebook-span guard compares `==`, and upstream's own condition has
  NOT been read.** Owner: W6. Issue
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

- **O17 — nothing in this row has yet LOADED a published DFlash2 artifact.**
  Owner: W6. Issue [#1314](https://github.com/mudler/vllm.cpp/issues/1314).

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


## Now

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

Next action: W6's G2/G3 on a leased GPU against the PR-head oracle — which also
carries O16 and O17 — then
`## Outcome`. **No throughput number is claimed by this wave and none is
admissible yet**: `## Gates` defers every ratio until G2 and G3 read, and a
DFlash2 draft is additionally off the paged CUDA-graph fast path, because the
selector needs the hidden states of the same forward its logits came from.
