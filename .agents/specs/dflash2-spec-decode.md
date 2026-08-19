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
  `src/vllm/entrypoints/model_loader.cpp::RefuseDflash2Draft`, cited by SYMBOL
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
| `qwen3_dflash2.py` conv | `vt::DFlashGroupedConv` op + `src/vllm/model_executor/models/qwen3_dflash2.cpp` | CPU reference first, CUDA after, as `KERNEL-ATTN-DFLASH-BLOCK` did |
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

## Owed

Everything here is landed-but-not-yet-reachable, or found and not fixed. Each
entry names what is missing, why it is not fixed where it was found, and the
wave that discharges it. AGENTS.md `## Nothing lands dead` permits a staged
slice to land unreached only when this list, the commit body and the pull
request body all name it, so this section is the record that permission depends
on and not a summary.

- **O1 — the `is_causal` half of W1 is INERT at its own merge commit, on both
  container arms.** Owner: this row, discharged by W2. Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314).
  `ResolveQwen3DFlashAttnModes`' top-level arm, `MakeQwen3DFlashDraftConfig`'s
  carry of the key, and `MakeDflashGgufConfig`'s read of
  `dflash.attention.causal` are all live code with unit gates, and no checkpoint
  this commit ADMITS can take any of them. The reason is the other half of the
  same wave: every artifact that declares the key also declares the DFlash2
  markers W1 refuses — `z-lab/Qwen3.8-27B-DFlash2` declares `is_causal false`
  beside `architectures: ["DFlash2DraftModel"]`, and
  `z-lab/Qwen3.8-27B-DFlash2-GGUF` declares `dflash.attention.causal` beside
  `dflash.selector_rank` — and `RefuseDflash2Draft` throws on both before any
  config is built. Every published DFlash1 artifact declares neither key, so it
  takes the legacy arm exactly as before, which is the inertness the wave's
  gates assert on purpose. This is `.agents/reachability.md`'s "unselected
  branch" shape: the branch is reached by construction in a test and by no input
  the production entry point accepts. It becomes live in W2, which is the wave
  that lifts the refusal for the parts it implements. W1 lands it anyway because
  splitting a refusal from the rule that makes the refused checkpoint correct
  would land the refusal alone and leave the rule to be rediscovered.
- **O2 — the loader's production call sites are not gated for the causality
  carry.** Owner: this row, discharged by W2. Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314). Mutation-proven by
  W1's fresh reviewer: appending `draft->config.raw.erase("is_causal");` after
  the `MakeQwen3DFlashDraftConfig` call in `LoadDflashDraft`
  (`src/vllm/entrypoints/model_loader.cpp`, safetensors arm) and after the
  `MakeDflashGgufConfig` call (GGUF arm) each COMPILES CLEAN and leaves both
  focused suites GREEN. What holds the carry is the direct unit call on the two
  builders, so the gate measures the builders and not the loader that uses them.
  It is not repaired where it was found because an entry-point gate on this path
  has to load a draft, and a draft load needs real draft weights: `LoadDflashDraft`
  reads the config and the shards in one function and shares `embed_tokens` and
  `lm_head` off a live target. W2 brings the DFlash2 fixture weights that make
  such a gate constructible; before then the only reachable form is another unit
  call, which is the thing that already fails to hold.
- **O3 — `MakeQwen3DFlashDraftConfig` cannot parse the published DFlash2
  `config.json` at all.** Owner: this row, discharged by W2. Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314). Found by W1's fresh
  reviewer and confirmed against the file on 2026-08-19: the builder does
  `c.at("rope_theta")` and `c.at("block_size")`, and `z-lab/Qwen3.8-27B-DFlash2`
  nests them as `rope_parameters.rope_theta` and `dflash_config.block_size`, with
  neither key present at the top level. Both `at` calls throw, so the builder
  cannot construct a config for the checkpoint this row exists to run. It is
  invisible today because W1 REFUSES that checkpoint earlier, at
  `RefuseDflash2Draft`, which runs before any config is built. It is a W2
  blocker rather than a W1 defect: the builder is only ever asked to parse a
  DFlash2 config once W2 lifts the refusal, and repairing it in W1 would land a
  parse arm for a shape nothing feeds. `transformers` moved RoPE settings under
  `rope_parameters`, so the fix is a fallback and not a replacement — DFlash1
  checkpoints still carry the flat spelling.
- **O4 — the `use_swa` causality repair (#1366) is UNREACHED, and the config
  builder cannot parse a draft that declares no `layer_types`.** Owner: this row,
  discharged by W2. Issue
  [#1314](https://github.com/mudler/vllm.cpp/issues/1314). Found by W1's second
  fresh review and repaired here as a RECORD correction rather than as code,
  because making it reachable is not a small and clear change. Three facts, each
  checked on 2026-08-19:
  (a) `MakeQwen3DFlashDraftConfig` does `c.at("layer_types")`, and upstream reads
  `getattr(config, "layer_types", None)` (`qwen3_dflash.py:134`, and `:66` in
  `_dflash_layer_causal`, @
  vllm-project/vllm#52816 head `19c9351904df4c63042671bc67a866ca48dc7d6f`), so an
  absent key is upstream's `None` and is this builder's raw
  `[json.exception.out_of_range.403]`. Mirroring that one `getattr` is a
  three-line change, and `layer_types` is the ONLY key of MiMo's real
  `dflash/config.json` this builder is missing -- every other name it reads is
  present at the top level there, which is what separates this entry from O3.
  Both halves of that are MEASURED rather than argued, on the published file
  (sha256 `2ed5a998f5f57e00a9fe14d2b3e767f06e49462a97eb09d80c927e112a585c9e`)
  driven through the production builder by a scratch program on 2026-08-19. As
  shipped it prints `THREW: [json.exception.out_of_range.403] key 'layer_types'
  not found`. With the three-line fallback applied in a scratch copy, restored
  byte-for-byte afterwards and verified by sha256, the same file builds and
  `ResolveQwen3DFlashAttnModes` answers five layers, every one `causal=0` with
  `sliding_window=1024` -- exactly upstream's `layer_types=None` + `use_swa=True`
  docstring row, and the opposite of what this engine answered before #1366. So
  the rule and the fallback are both right; what is missing is anything that can
  feed them.
  (b) It would still buy no reachability. `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash`
  is the only published draft with `use_swa` and no `layer_types`, and its target
  `MiMoV2ForCausalLM` is `INVENTORIED`, unassigned and unimplemented in
  `.agents/model-matrix.md`, so no production entry point can serve the model
  that draft heads. A drafter is not reachable without its target.
  (c) It would make a WRONG path selectable where a loud parse failure
  stands today.
  Upstream reads `dflash_config.attention_sink_bias` and passes a per-head sink
  bias into its `Attention` (`qwen3_dflash.py:309-313` and `:240-257` @ that
  head); MiMo's config sets it true and this lane has no attention sink at all,
  so a MiMo draft that parsed would load with the sinks silently absent --
  acceptance-only and token-invisible, the exact class this row exists to remove.
  Landing (a) alone converts a loud parse error into a quiet wrong answer.
  W2 discharges this with O3, and it must land the `layer_types` fallback, a
  named refusal for `dflash_config.attention_sink_bias`, and the loader-entry
  gate O2 owes, as one change rather than three.

## Now

`SPEC-DFLASH2` is `ACTIVE`. W1 landed on 2026-08-19: the `DFlash2DraftModel`
route and D4's `is_causal` precedence, both red-first and both mutation-proven on
CPU. No DFlash2 mechanism landed with it, and none is claimed.

What W1 ships is a REFUSAL. A draft whose `config.json` declares
`DFlash2DraftModel` is refused at startup, before any weight is read, with both
missing mechanisms named — from the dflash branch of
`LoadedEngine::ResolveSpecConfig` and again at the top of
`LoadedEngine::FromModelDir`, which is the site that matters, because the dflash
draft load runs there BEFORE the constructor's resolution. Refusing rather than
loading is the whole point: a DFlash2 checkpoint carries DFlash1's entire tensor
set, so the DFlash1 lane would load it with nothing missing and draft worse
tokens with no visible symptom.

D4 landed with it, in both halves, and is NOT YET REACHED. `ResolveQwen3DFlashAttnModes`
resolves a top-level `is_causal` ahead of `dflash_config.causal` and ahead of the
legacy `layer_types` rule, and `MakeQwen3DFlashDraftConfig` — moved out of the
loader's anonymous namespace so the key it carries is gateable at all — copies the
key off the draft's own `config.json`. A resolution that reads a key the config
builder drops is half a port, and only the two together make the rule reachable.
Reachable in principle: no checkpoint W1 ADMITS declares the key, because every
artifact that declares it also declares the DFlash2 markers the same commit
refuses. `## Owed` O1 records that precisely, O2 records that the loader's own
call sites are not gated for the carry, and W2 discharges both. What W1 asserts
about D4 is therefore a unit guarantee plus the inertness of the DFlash1 lane,
and no more than that.

W1 also covers the GGUF drafter, which the spec's own W1 text did not name and
which the `config.json`-keyed classification cannot see. `z-lab/Qwen3.8-27B-DFlash2-GGUF`
@ `57ab3265056d4024870b0621cfc2c127537020ed` writes `general.architecture = "dflash"`,
byte-identical to a DFlash1 drafter, and a GGUF carries no `architectures` array
at all — so `qwen3_dflash_gguf.cpp` would have loaded it as DFlash1 with no
error. The discriminator is therefore the DFlash2-only metadata
(`dflash.selector_rank`, `dflash.selector_top_k`, `dflash.conv_kernel_size`), and
`dflash.attention.causal` is the GGUF spelling of `is_causal` and resolves in the
same precedence. Both were read off the published file on 2026-08-19; the shipped
DFlash1 drafter `muse-glimmer-30b-gguf/dflash-kquant.gguf` carries none of those
keys and is unchanged. This is a REFUSAL on the GGUF axis, not the GGUF drafter
ARM, which stays W5.

W1's fresh review also raised [#1366](https://github.com/mudler/vllm.cpp/issues/1366)
against the same function, and it is FIXED IN FLOW rather than deferred, per
AGENTS.md. It is pre-existing and it is D4's failure class one arm over. The
legacy fallback read the RESOLVED `is_sliding`, which `dflash_config.use_swa`
forces true on every layer; upstream reads the DECLARED `layer_types`
(`bool(layer_types) and layer_types[i] == "sliding_attention"`,
`qwen3_dflash.py:66-67` @ the PR head), and states the consequence as a row of
its own `_resolve_layer_attention` docstring table -- `layer_types=None` +
`use_swa=True` -> causal False -- naming `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash`
as the published checkpoint of that shape. Such a DFlash1 draft therefore ran
every layer CAUSAL here and non-causal upstream, with the verify lossless and
only acceptance moving. It went uncaught because upstream's parametrize table
has no `use_swa` row either, so the ported cases were faithful to the ported
table and silent about the arm. The same issue's second half is the coercion:
`is_causal` was honoured only as a JSON boolean, while upstream tests presence
and coerces, and the GGUF arm's `KvI64` already took every integer width -- so
`"is_causal": 0` fell through in silence and the two containers disagreed with
each other. Both halves are repaired red-first and mutation-proven, and
NEITHER IS REACHED. An earlier revision of this section, and `4941dfbfe`'s commit
body, claimed the `use_swa` half was -- "REACHED today, because the checkpoints it
governs are DFlash1 ones this engine already admits". That was wrong on two
independent counts, both established by W1's second fresh review on 2026-08-19 and
confirmed against the published files and against this tree.
`XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` is the only published draft of the governed
shape, and it declares NO `layer_types`, while `MakeQwen3DFlashDraftConfig` does
`c.at("layer_types")` and throws `[json.exception.out_of_range.403] key
'layer_types' not found` before any causality is resolved. Its target
architecture `MiMoV2ForCausalLM` is `INVENTORIED` and unassigned in
`.agents/model-matrix.md`, so this engine cannot serve the model that draft
heads, whatever the builder does. The GGUF arm cannot reach the rule either:
`MakeDflashGgufConfig` never writes `use_swa` and always fills `layer_types` from
the sliding-window pattern. The three other published DFlash1 drafts
(`z-lab/Qwen3.6-27B-DFlash`, `z-lab/Qwen3.5-9B-DFlash`,
`z-lab/gemma-4-31B-it-DFlash`) all declare `layer_types` and no `use_swa`, so the
repair leaves their resolution byte-for-byte unchanged -- the inertness half,
which is the only thing W1 asserts about either half. `## Owed` O4 records the
gap, names W2 as its owner, and states why the reachability repair was not
attempted in this flow.

Next action: W2, the grouped dynamic convolution, CPU reference first, against
the checkpoint's real shapes (taps 2, group 16, block 8). W2 must also discharge
`## Owed` O1, O2, O3 and O4, and O3 and O4 are blockers rather than cleanups.
