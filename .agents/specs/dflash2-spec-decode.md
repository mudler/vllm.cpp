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
`19c9351904df4c63042671bc67a866ca48dc7d6f`, base `9842d701`, 755+/5-, 11 files,
opened 2026-08-18, plus the stacked LM-head guard fix
[#52883](https://github.com/vllm-project/vllm/pull/52883), opened 2026-08-19.
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
| `qwen3_dflash2.py` selector | same file, plus a `vt` lattice op | `_score_edges` is one einsum; it is not the cost |
| `_topk` / FlashInfer radix | extend `src/vt/cuda/cuda_sample.cu:297-506` to emit pairs | D2 |
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
  reach it. G5 lands here.
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
- **D9 — the row gates the OUTPUT SCALARS against a checkpoint that SETS them,
  not against defaults.** Discovered after this spec landed
  ([#1327](https://github.com/mudler/vllm.cpp/issues/1327)); the brief that
  proposed it called it D8, which was already taken by the `is_causal` coercion
  divergence above, so it is D9 here. A port that reads all three with
  `.get(key, default)` passes every gate built from the Qwen3.8 draft alone,
  because that draft sets none of them — such a gate measures the default path
  and reports it as coverage. W3 owns the scalars and must gate them on
  `z-lab/Muse-Glimmer-30B-DFlash2`'s values.
- **D10 — the DFlash2 refusal MOVES from startup to the first propose, on the
  safetensors arm only.** W2 decision, 2026-08-19. W1 refused a
  `DFlash2DraftModel` draft before any weight was read, when both mechanisms were
  missing. W2 implements one of them, and keeping the startup refusal would leave
  every line of it unreachable from any production entry point — AGENTS.md
  `## Nothing lands dead` — with the conv gated only by tests that construct it,
  which is `.agents/reachability.md`'s test-only driver. So a safetensors DFlash2
  draft is now ADMITTED: it loads its conv weights, runs the conv in all three
  layer bodies, and is refused BY NAME at `RefuseDflash2CandidateSelector`, after
  the block forward and before anything samples. Cost: the refusal arrives at the
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
  target's `embed_tokens` and `lm_head`. W4 brings the speculator wiring that
  makes such a gate constructible. The consequence if it regressed is bounded and
  named: the conv would mask its taps against the checkpoint's DEFAULT block
  instead of the resolved `k`, which is invisible unless the two differ — and
  acceptance-only, token-invisible, when they do.
- **O6 — the CUDA arm of `vt::DFlashGroupedConv` is UNVERIFIED.** Owner: this
  row, discharged by the operator's GPU lease before W6. Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314). The kernel and its
  registration are written and reviewed
  (`src/vt/cuda/cuda_ops.cu`, `DFlashGroupedConvKernel` /
  `DFlashGroupedConvKernelCuda`), and the CUDA==CPU bit-identity case exists and
  is written to run
  (`tests/vt/test_ops_dflash2_grouped_conv.cpp`, six shapes covering both
  published blocks in bf16 and the modulo arm in f32). It has NEVER COMPILED:
  the authoring host has no `nvcc`, so the CUDA case reports
  `no CUDA backend; skipping CUDA dflash2-grouped-conv parity` and every one of
  the file's assertions runs on CPU. Two specific things are unproven rather than
  merely unrun: that the kernel compiles at all, and that
  `__fadd_rn`/`__fmul_rn` plus `ResRound` reproduce the CPU reference BIT-FOR-BIT
  on the f32 arm, where the intrinsics are the only thing forbidding an FMA
  contraction the CPU build pins off. This is named here rather than reported as
  a pass.

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
- **O7 — NO production call site of the selector refusal is gated.** Owner: this
  row, discharged by W4. Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314).
  `RefuseDflash2CandidateSelector` is called from exactly two places, and only
  one of them is production:

  - `GPUModelRunner::propose_drafts_block` (`src/vllm/v1/worker/gpu/runner.cpp`)
    — the PRODUCTION site, and NOT gated. Mutation-proven: deleting this call
    leaves all four focused suites GREEN.
  - `DflashProposeBlock`
    (`src/vllm/v1/worker/gpu/spec_decode/dflash/speculator.cpp`) — gated
    (deleting the call turns `test_qwen3_dflash2_draft` red, 1 case), but
    `DflashProposeBlock` has NO caller outside `tests/` at this commit. An
    exhaustive grep finds only its definition, its declaration, two prose
    comments in `runner.cpp`, and tests. This site is TEST-ONLY.

  W2's own record used to say the refusal "has TWO production call sites", one of
  them gated. That was wrong in the direction that flatters: production coverage
  of the refusal is ZERO, not one of two. AGENTS.md `## Nothing lands dead`
  grants the staged-slice exception only when this list names what is unreached,
  so an inaccurate entry here is a defect in the permission and not a wording
  problem. It is corrected rather than annotated.

  **Why it is not gated where it was found.** Reaching the runner site means
  reaching `propose_drafts_dflash`, which returns early unless `dflash_weights_`
  is set, and that member is only ever set on the `LoadedModel` construction
  path; the synthetic-weights `GPUModelRunner` constructors take no
  `SpeculativeConfig` at all. A gate therefore needs an on-disk TARGET plus an
  on-disk draft driven through the loader, a step that captures the target's aux
  multi-tap, and a populated per-request device KV store. That is the harness O5
  is already waiting on, and the one W4 builds when it wires the DFlash2
  speculator.

  **What this does NOT put in doubt.** The grouped convolution is production-
  reached and mutation-detected. Each conv call site was deleted separately —
  `attention_conv` and `mlp_conv`, in each of the three layer bodies, six
  mutations over twelve calls — and every one turned `test_qwen3_dflash2_draft`
  red, including through `ForwardPagedBody`, which is what
  `ForwardBlockLogitsWithDeviceKV` and therefore the production decode path
  reaches.
- **O8 — the context-KV precompute applies NO convolution, and nothing says
  whether that stays equivalent.** Owner: this row, answered by W3 or W4. Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314). Raised by the wave's
  second fresh reviewer, and recorded rather than fixed because answering it is a
  design question about the engine's context path rather than a repair to what W2
  shipped.

  `Qwen3DFlashModel::PrecomputeContextKVDevice`
  (`src/vllm/model_executor/models/qwen3_dflash.cpp:149`) projects EVERY layer's
  context K/V from one shared `hidden_norm(context_states)`, and applies no
  convolution at any layer. Upstream has no analogue of this precompute: its
  context K/V is whatever the earlier block forwards wrote, and under DFlash2
  those forwards wrote from a CONV'd stream. Ours is projected from an
  unconvolved shared tensor; upstream's came from a convolved per-layer one.

  The shortcut predates DFlash2 and is correct for DFlash1, where no conv exists.
  Whether it stays equivalent now that the conv does is not addressed anywhere in
  this spec. If it is wrong, the symptom is the defect class this row exists to
  remove: acceptance-only and token-invisible, because the verify is lossless and
  the engine still emits the target's tokens. W3 touches this path when it lands
  the selector, and either shows the two agree or replaces the precompute.

## Now

`SPEC-DFLASH2` is `ACTIVE`. W1 landed on 2026-08-19 (the route and D4's
`is_causal` precedence). **W2 landed on 2026-08-19: the grouped dynamic depthwise
convolution, REACHED.**

**What W2 ships is a mechanism, where W1 shipped a refusal.** `vt::DFlashGroupedConv`
is the project's first grouped dynamic depthwise convolution:
`out[i,c] = sum_t (base[side,t,c] + delta[i,side,t,g(c)]) * x[i-t,c]`, with tap
`t` contributing only where `(i mod block) >= t`, `g(c) = c / group_size`, and
`base_kernel` dim 0 the prepare/finish SIDE rather than a tap. The CPU kernel is
the authoritative reference and rounds to the tensor dtype after each step, as
upstream's bf16 chain materializes it — which is what lets the CUDA mirror be
specified BIT-IDENTICAL rather than within an envelope. That per-step rounding
is now itself gated, on CPU and in bf16, which is the only arm where it is
observable at all: see O6 for what the wave's second review found and what it
cost. Both of upstream's position-mask arms are ported (`pos & (block-1)` and
`pos % block`) and gated at block 5, 8 and 16.

**The refusal MOVED so that the conv could be reached.** A safetensors
`DFlash2DraftModel` draft is now admitted at `CheckDflash2DraftArm`, loads its
`attention_conv`/`mlp_conv` tensors through `LoadQwen3DFlash`, runs the conv in
ALL THREE of the draft's layer bodies — `ForwardBlockLogits`,
`ForwardWithCtxKVDev` and `ForwardPagedBody`, the last being what the production
decode path reaches through `ForwardBlockLogitsWithDeviceKV` — and is then
refused BY NAME at `RefuseDflash2CandidateSelector`, after the block forward and
before anything samples. `## Risks/decisions` D10 carries the decision and its
cost; a startup NOTICE names the boundary so the later refusal is not a surprise.
The GGUF arm keeps its startup refusal and moves with W5.

**Reachability was measured, not argued, and it cost two gate repairs.** The
first version of the model-level gate could not tell one missing call site from
none: it activated both convs at once, so deleting only the context-aware body's
`attention_conv` left the suite GREEN. The second could not see the SIDE: forcing
`args.side` to 0 in the kernel left the model suite GREEN. Both were found by
running the mutation rather than by reading the test, and both were repaired
before the wave landed — each conv is now driven ALONE through each body, and the
two sides are separated by `base_kernel[side]` scalars against a common identity
baseline. It cost a third repair after the wave's second review, on the same
pattern: the PER-STEP ROUNDING had no executing assertion, because every case in
the op suite ran in f32 where that rounding is the identity (O6). The final
mutation set turns the focused suites red for: each body's call sites (three
separate mutations), the side index, the block mask, the group map, the per-step
bf16 rounding, the `rope_parameters` fallback, the `dflash_config.block_size`
fallback, the `layer_types` fallback, the `attention_sink_bias` refusal, the
uniform-block guard, the `DflashProposeBlock` refusal call, and restoring W1's
startup refusal. It does NOT turn them red for the refusal's production call
site, which is O7.

**Four `## Owed` entries are discharged and four are new.** O1 (the `is_causal`
rule was inert), O2's weight half, O3 (`MakeQwen3DFlashDraftConfig` could not
parse either published DFlash2 config) and O4 (`layer_types`, plus the
`attention_sink_bias` refusal that had to land with it) are closed. O5 records
that `LoadDflashDraft`'s own `conv_block_size = k + 1` is UNGATED and
mutation-proven so; O6 records that the CUDA arm has never compiled on this host
and is owed to a GPU lease, and what the CPU-side rounding gate does and does not
now prove; O7 records that NO production call site of the selector refusal is
gated — zero, not one of two, which is what the entry said before the wave's
second review measured it; O8 records that the context-KV precompute applies no
convolution and that nobody has shown the shortcut stays equivalent now that the
convolution exists. None of the four is a claim wearing a pass.

**#1327 is corrected in this wave.** `## Upstream chain` said no published
checkpoint exercised `input_embedding_scale`, `output_multiplier` or
`final_logit_softcapping`. `z-lab/Muse-Glimmer-30B-DFlash2` sets
`output_multiplier 0.19611613513818404` and `final_logit_softcapping 20.0`, and
ships `block_size` 16 against the 27B's 8. Both scalars are applied to candidate
VALUES before the selector scores them, so a wrong one reorders the top-K and
moves acceptance without raising. `## Scope`'s exclusion of "a second DFlash2
target family" is dropped (upstream registers ONE class and both checkpoints
declare `model_type` `qwen3`), `## Gates` G1 now requires both block shapes, and
`## Risks/decisions` D9 records that the scalars must be gated against the
checkpoint that sets them rather than against defaults.

**The admitted checkpoints are now PINNED, and their hashes are ours.**
`docs/USAGE.md` gains a `## DFlash2 drafts: the exact checkpoints` table: repo,
revision, file, byte count and sha256 for the admitted bf16 safetensors draft
(`z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c`) and for all three refused GGUF arms,
plus the target the draft heads and the second published draft's revision. Every
sha256 was computed over a local copy rather than read from a hub API, because an
unauthenticated tree API can return an `lfs.oid` that hashes nothing; the
safetensors shard was also checked semantically (81 tensors, all BF16, last data
offset exactly on the file size). The same section says what the gate actually
reads, which is not those bytes: the published `config.json` documents embedded
byte-for-byte in the test, and a safetensors file the test WRITES with the
published tensor names.

Next action: W3, the candidate selector — the lattice op, the codebooks in the
loader, and the top-k that EMITS pairs (D2). It is the wave that lifts the
refusal W2 leaves behind, and D9 binds it to Muse Glimmer's scalars.
