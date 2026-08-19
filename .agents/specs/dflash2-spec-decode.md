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

Out of scope: any change to the DFlash draft's own behaviour beyond the shared
`is_causal` rule; a second DFlash2 target family (upstream registers exactly one,
Qwen3); the DSpark lane; and any throughput claim, which `## Gates` defers with
its reason.

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
`output_multiplier`, `final_logit_softcapping` — are ABSENT from this config and
take 1.0, 1.0 and disabled. No published checkpoint exercises them, so the port
implements them and gates them synthetically rather than claiming checkpoint
coverage.

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
  (`src/vllm/model_executor/models/qwen3_dflash_weights.cpp:56-60,155-157`). The
  class comment at `include/vllm/model_executor/models/qwen3_dflash.h:75-77`
  claims the draft owns both; it is STALE and this row corrects it.
- `src/vt/cuda/cuda_sample.cu:297-506` is a sort-free block-cooperative
  pivot-bracket threshold search, one block per row, ported from the same
  FlashInfer `TopK/TopPRenormProb` approach the selector's top-k uses.
- `Qwen3DSparkModel::SampleSequentialDevice` is the shipped precedent for a
  sequential per-step draft walk that runs on device rather than on the host.

Absent, and owed by this row:

- No route for `DFlash2DraftModel`. `SpeculativeConfig` classifies a DSpark draft
  from its own `config.json` (`include/vllm/config/speculative.h:134-158`) and has
  no DFlash equivalent for a second architecture. The identical gap for
  `DSparkDraftModel` is open as [#1193](https://github.com/mudler/vllm.cpp/issues/1193);
  this row does not fix that one, and must not collide with it.
- No grouped dynamic convolution anywhere in `vt`.
- No candidate selector, no lattice, no path walk.
- No top-k that EMITS the surviving (id, value) pairs. The threshold search
  above masks below the k-th largest and returns no indices.
- `is_causal` is not read. `include/vllm/model_executor/models/qwen3_dflash.h:22-24`
  resolves causality by the legacy rule alone.

## Port map

| Upstream | Ours | Note |
|---|---|---|
| `qwen3_dflash2.py` conv | `vt::DFlashGroupedConv` op + `src/vllm/model_executor/models/qwen3_dflash2.cpp` | CPU reference first, CUDA after, as `KERNEL-ATTN-DFLASH-BLOCK` did |
| `qwen3_dflash2.py` selector | same file, plus a `vt` lattice op | `_score_edges` is one einsum; it is not the cost |
| `_topk` / FlashInfer radix | extend `src/vt/cuda/cuda_sample.cu:297-506` to emit pairs | D2 |
| `dflash2/speculator.py` walk kernel | `src/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.cpp` + a CUDA walk kernel | D3: device from day one |
| `dflash2/speculator.py` draft-logit cache | same file | the realized q the rejection sampler reads at T>0 |
| `registry.py` + `spec_decode/__init__.py` | `include/vllm/config/speculative.h`, `src/vllm/entrypoints/model_loader.cpp` | classify from the draft's own config, as `IsDsparkDraft` does |
| `_dflash_layer_causal` | `include/vllm/model_executor/models/qwen3_dflash.h:22-24` | D4; the ONE shared-behaviour edit |
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
  checkpoint's real shapes (taps 2, group 16, block 8, K 16, rank 256).
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

## Now

`SPEC-DFLASH2` is `READY`. The spec is committed and base-reachable, which is
what a helper dispatch needs; no production code has landed. Next action: W1,
dispatched to a fresh implementer against this spec, red-first on the causality
gate.
