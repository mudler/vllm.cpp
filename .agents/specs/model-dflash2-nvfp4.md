# MODEL-DFLASH2-NVFP4 — the ModelOpt NVFP4 arm of the DFlash2 draft loader

Row: `MODEL-DFLASH2-NVFP4`
Issues: [#2758](https://github.com/mudler/vllm.cpp/issues/2758)
Base SHA: `4e748e4a7` (`origin/main`)
Parent rows: [`SPEC-DFLASH2`](dflash2-spec-decode.md) is the draft itself;
[`MODEL-DFLASH2-EXL3`](model-dflash2-exl3.md) is the arm this one mirrors in
shape; [`QUANT-QWEN38-27B-NVFP4-ARM`](qwen38-27b-quant-arms.md) owns the
ModelOpt NVFP4 reader this row consumes.
Consumer: [`BENCH-QWEN38-27B-SOTA`](bench-qwen38-27b-nvfp4-matched.md), leg F.

Upstream pin: vLLM `e126687a9a828d513c01a07cd69f025f27d63280` (the active parity
pin, [upstream-sync.md](../upstream-sync.md)). vLLM defines every line of this
behaviour, so nothing here is a product decision and nothing was asked.

## Now

`ACTIVE`. The loader-side arm is written and gated hermetically ON THE MERGED
TREE -- `origin/main` moved 42 commits under this branch and the numbers below
are from after the merge, not carried forward from the tree the branch was cut
from: `test_qwen3_dflash2_nvfp4` 15/15 test cases, 334/334 assertions, with the four
neighbouring draft suites (`test_qwen3_dflash2_exl3` 5/5,
`test_qwen3_dflash2_draft` 44/44, `test_dflash_causality` 13/13,
`test_dflash2_draft_routing` 12/12) and the six dense NVFP4 suites the exported
reader touches all unchanged and green. The red before it was the issue's own
sentence, verbatim: `vt: qwen3_dflash: expected BF16 for
layers.0.self_attn.q_proj.weight at
src/vllm/model_executor/models/qwen3_dflash_weights.cpp:50`. No device has run
it; see [Owed](#owed).

## The gap, verified against the tree rather than taken from the issue

`LoadQwen3DFlash` asks the arm question exactly once and in one direction
(`src/vllm/model_executor/models/qwen3_dflash_weights.cpp`):

```cpp
const bool exl3 = static_cast<bool>(has) && dense_loaders::IsExl3Projection(has, "fc");
```

There is no third rung, and `MakeQwen3DFlashDraftConfig` copies named keys of
which `quantization_config` is not one — so the draft's own declaration is not
merely unused, it never reaches the loader at all.

**The failure the issue describes is real and reproduces by reading.** `fc` is
excluded from quantization in this artifact and is therefore BF16, so
`IsExl3Projection(has, "fc")` answers false, the BF16 reader is selected, `fc`,
`hidden_norm` and `norm` all load, and the load then dies inside layer 0 at
`ConcatRawNK` → `LoadBf16Direct`:

```
vt: qwen3_dflash: expected BF16 for layers.0.self_attn.q_proj.weight
```

Nothing in that sentence says NVFP4, ModelOpt, or "this build has no arm for a
quantized draft".

### The artifact, read from its own bytes

`maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal` @
`bd7a934213c47a9e7ef69eef36bb3325f47fd1f1`, six files, `model.safetensors`
1,550,153,248 B. The safetensors header (19,200 B, HTTP range read
2026-09-05) carries **186 tensors**, and `config.json` and the sidecar
`hf_quant_config.json` carry the same quantization document twice.

| | value |
|---|---|
| `architectures` | `["DFlash2DraftModel"]` |
| `quantization_config.quant_method` | `modelopt` |
| `quantization_config.quant_algo` | `NVFP4` — **W4A4**, not `W4A16_NVFP4` |
| `group_size` | 16 |
| `kv_cache_quant_algo` | `FP8` |
| `exclude_modules` | 12 entries, all EXACT module names, no wildcard |
| `num_hidden_layers` | 5 |
| `hidden_size` / `intermediate_size` | 5120 / 17408 |
| `is_causal` | `false`; `layer_types` all `sliding_attention` |
| `rope_parameters.rope_theta` | 10000000 (nested spelling only) |

**35 modules are quantized** — 7 per layer × 5 layers — and each ships exactly
four tensors:

| operand | dtype | shape (`layers.0.self_attn.q_proj`) |
|---|---|---|
| `.weight` | `U8` | `[4096, 2560]` = `[N, K/2]` |
| `.weight_scale` | `F8_E4M3` | `[4096, 320]` = `[N, K/16]` |
| `.weight_scale_2` | `F32` | `[]` scalar |
| `.input_scale` | `F32` | `[]` scalar |

That is the **ModelOpt spelling** (`.weight` + `.weight_scale_2`), not the
compressed-tensors one (`.weight_packed` + `.weight_global_scale`), which is
why `dense_loaders::IsCtNvfp4Projection` would have missed it.

**The 12 excluded modules are BF16 and are exactly the ones this engine has no
packed owner for:**

| excluded module | shape | who reads it |
|---|---|---|
| `fc.weight` | `[5120, 25600]` | `Qwen3DFlashWeights::fc` |
| `candidate_selector.hidden_projection.weight` | `[256, 5120]` | `Dflash2SelectorWeights::hidden_projection` |
| `layers.N.attention_conv.kernel_projection.weight` | `[1280, 5120]` ×5 | `Qwen3DFlashConvWeights::kernel_projection` |
| `layers.N.mlp_conv.kernel_projection.weight` | `[1280, 5120]` ×5 | same, MLP side |

Every norm, both selector codebooks and both `base_kernel` tensors are BF16 as
well, and the file ships **no `lm_head` and no `embed_tokens`** — the draft
shares the target's, exactly as the EXL3 and BF16 drafts do.

**Q, K and V are three separate modules with three separate `weight_scale_2`
scalars.** There is no merged `qkv_proj` operand in the file, and none in the
target's own ModelOpt path either: `LoadDenseAttn`'s `load_projection`
(`src/vllm/model_executor/models/qwen3_5_dense_weights.cpp`) reads q/k/v/o as
four independent `Nvfp4Weight`s and `LoadDenseMlp` reads gate/up/down as three.
So the seven-owner shape below is the tree's own established NVFP4 shape and not
a concession this row invents.

## Scope

**In.** The loader-side ModelOpt NVFP4 arm for the DFlash2 draft: the draft's
own `quantization_config` reaching the loader, the arm question, the seven
per-layer packed owners, the exclusion rule, a refusal that names the missing
part for every module this build has no packed owner for, the forward routing
through the shared `layers::MakeLinearMethod` / `MakeMlpGateUpMethod` seam, and
a hermetic gate that enters through the production loader.

**Out.** Any GPU measurement (no lease was taken for this row, by developer
direction). The compressed-tensors NVFP4 spelling on a draft — no published
DFlash draft ships it, so an arm for it would land unreached. The W4A4
activation path (owned by `QUANT-QWEN38-27B-NVFP4-ARM`, [#2760](https://github.com/mudler/vllm.cpp/issues/2760)).
The FP8 KV-cache arm (owned by `KV-FP8`, [#1593](https://github.com/mudler/vllm.cpp/issues/1593)).

## Design

### D1. The declaration reaches the loader, because upstream reads it

`DFlashQwen3Model.__init__` resolves its quantization from
`get_draft_quant_config(vllm_config)` and hands the result to every `Linear` it
builds — `qkv_proj` and `o_proj` at `qwen3_dflash.py:216-230`, the decoder
layer's two sublayers at `:339` and `:348`, `fc` at `:456-464`, all reading
`self.quant_config` set at `:410`.
`get_draft_quant_config` is `vllm/model_executor/models/utils.py:929-948`, and
its whole point is stated in its own docstring: *"Draft models should use their
own quantization config instead of the verifier/target model's config."* It
returns `VllmConfig.get_quantization_config(draft_model_config, draft_load_config)`,
which is the DRAFT's `config.json`.

So `MakeQwen3DFlashDraftConfig` carries `quantization_config` into `cfg.raw`.
That is one added key on a builder that already copies `dflash_config`,
`block_size` and `is_causal` for the same reason: a key the builder drops is a
key the resolution can never see.

### D2. The arm question is the DECLARATION, and the tensors are cross-checked

Upstream never probes tensor names for this. Each `Linear` asks
`quant_config.get_quant_method(layer, prefix)`, and `ModelOptQuantConfigBase.get_quant_method`
(`modelopt.py:177-215`) consults `is_layer_excluded(prefix)` (`:139-175`) FIRST
and returns `UnquantizedLinearMethod()` for an excluded `LinearBase`, then the
NVFP4 linear method for everything else. That is the rule this arm mirrors.

Routing by the declaration rather than by a name probe is also forced by this
artifact: `fc` is BF16 here, so the EXL3 rung's "ask once, on `fc`" cannot be
reused, and asking `layers.0.self_attn.q_proj` instead would key the arm on a
name that upstream does not consult.

`IsLayerExcluded`'s three passes are mirrored in order — exact match, then the
legacy substring rule kept for pre-0.39 ModelOpt exports, then `fnmatch`
wildcards. The wildcard half reuses `layers::modelopt::detail::FnMatch`, which
already mirrors `fnmatch` for `MixedPrecisionConfig`; a second implementation of
one rule is the "two descriptions" failure `AGENTS.md` §"Changing the rules"
names. The `packed_modules_mapping` half of upstream's `is_layer_skipped` is
NOT mirrored, because this loader builds no fused module out of separately
excluded shards and the draft declares no mapping — stated here rather than left
to be discovered.

**The tensors are still read, as a CROSS-CHECK in both directions**, which is
the polarity `QUANT-QWEN38-27B-NVFP4-ARM` W5 set for the target:

- declared quantized (not excluded) but shipping no NVFP4 operands → refuse, by
  module name;
- declared excluded but shipping NVFP4 operands → refuse, by module name;
- shipping NVFP4 operands with no declaration at all → refuse, by module name,
  naming the absent `quantization_config`.

Each of the three is a real mis-load that no later check can see: an NVFP4
module read as BF16 dies on a dtype (loud), but a BF16 module read as NVFP4, or
a quantized module silently skipped, produces a correctly shaped and completely
wrong weight, and the draft would still emit the target's tokens because the
verify is lossless. Only acceptance would move.

### D3. Seven owners per layer, and no merge

`Qwen3DFlashLayerWeights` gains seven `Nvfp4Weight`s, exactly paralleling the
seven `Exl3Weight`s beside them, and `qkv_proj` / `gate_up_proj` stay empty on
this arm.

Merging is not deferred for convenience; it is wrong for this file. Each of the
three q/k/v modules carries its own `weight_scale_2`, and merging along output
rows requires taking `max()` over the shards' divisors and rescaling every group
scale (which is what `dense_loaders::LoadMergedCtNvfp4W4A16` does for
compressed-tensors, where a merged operand is what the producer emits). This
producer emitted three tensors and three scalars, and the target's own ModelOpt
path in this tree reads q/k/v/o unmerged for the same reason. `MergedQkvEnabled()`
is therefore not asked on this arm, exactly as it is not asked on the EXL3 one.

### D4. The four modules with no packed owner are REFUSED BY NAME

`fc`, `candidate_selector.hidden_projection` and the two conv
`kernel_projection`s have no `Nvfp4Weight` owner in this engine, and this
artifact excludes all four, so an owner for them would land unreached — the
shape `AGENTS.md` §"Nothing lands dead" refuses. They are refused by name
instead, at the arm question rather than 400 lines later, with the module, the
declared algorithm, the owner that is missing, and the owning row and issue in
the message.

This is the half the issue calls "what makes the missing capability visible
instead of discoverable", and it is what `AGENTS.md` §"Shared seams" requires of
an unimplemented arm.

### D5. The forward routes through the shared seam, not a parallel path

`layers::MakeLinearMethod(const OwnedTensor&, const Nvfp4Weight&)` and
`layers::MakeMlpGateUpMethod(...)` already exist
(`include/vllm/model_executor/layers/quantization/compressed_tensors/schemes/nvfp4.h`)
and are what the target's dense forward binds. The draft's three forward bodies
bind the same factories through two file-local selectors that answer
`exl3 → nvfp4 → bf16` in one place, so no body carries its own copy of the
three-way choice — the defect `#2202` records for the merged-qkv seam and the
reason `ProjectDflashQkv` exists at all.

### D6. The declared-W4A4-executed-W4A16 divergence is SAID, not silenced

All 35 quantized modules ship `input_scale` and the declaration is `NVFP4`, not
`W4A16_NVFP4`. `LoadNvfp4AnyNaming` consumes `input_scale` only under
`VT_MODELOPT_W4A4=1`, which defaults to 0, so this draft runs W4A16 against a
W4A4 declaration — the same divergence [#2760](https://github.com/mudler/vllm.cpp/issues/2760)
records for `RadixArk/Qwen3.8-27B-NVFP4`, on the drafter.

The draft loader prints its own one-line notice naming both algorithms, the
count and the knob. It does NOT reuse `layers::modelopt::ActivationArmNotice`,
which is scoped to `MIXED_PRECISION` configs and answers `""` on a plain
`NVFP4` one; widening that function is #2760's change and moves a gate model's
arm, so it is not made here. The notice does not refuse and moves no arm.

### D7. `kv_cache_quant_algo: "FP8"` is NOT a divergence at default settings

Checked at the pin rather than assumed. `BaseKVCacheMethod.process_weights_after_loading`
gates every scale it applies on `is_quantized_kv_cache(layer.kv_cache_dtype)`
(`vllm/model_executor/layers/quantization/kv_cache.py:95-140`), and
`kv_cache_dtype` comes from `cache_config.cache_dtype`, which is the
`--kv-cache-dtype` request and defaults to `auto`. The declaration alone does
not make upstream run an fp8 KV cache, and this artifact ships **zero**
`k_scale` / `v_scale` tensors, so upstream's own default is bf16 KV — which is
what this engine runs. Recorded as a checked non-divergence; the arm itself
stays owed by `KV-FP8` ([#1593](https://github.com/mudler/vllm.cpp/issues/1593)).

## Tests

`tests/vllm/models/test_qwen3_dflash2_nvfp4.cpp`, target
`test_qwen3_dflash2_nvfp4`. Every case enters at
`vllm::MakeQwen3DFlashDraftConfig(config_json)` followed by
`vllm::LoadQwen3DFlash(shards, config, num_taps, mask_id)` — the two calls
`LoadDflashDraft` makes in that order from the `--speculative-config` path
(`src/vllm/entrypoints/model_loader.cpp`), over a REAL safetensors file written
to a temp dir. No case constructs a loader or a weight struct by hand.

The fixture mirrors the published artifact's shape at 1/40th scale: 5 → 2
layers, H 5120 → 128, I 17408 → 256, and the same 7-quantized-plus-12-excluded
split per layer. `K` is a multiple of 16 on every quantized module, which is
what NVFP4 group-16 requires.

| case | what it falsifies |
|---|---|
| `loads the published shape` | the arm exists at all; all 7 owners populated per layer with the right `n`, `k` and `scale2`; `qkv_proj`/`gate_up_proj` empty; the 12 excluded modules read BF16 |
| `unquantized draft is byte-unchanged` | inertness: no `quantization_config` → the existing BF16 path, `Nvfp4Weight`s all empty |
| `fc quantized is refused by name` | D4: the message names `fc`, NVFP4 and the missing owner, and is not a BF16 dtype complaint |
| `selector projection quantized is refused by name` | D4, second module |
| `conv kernel_projection quantized is refused by name` | D4, third module |
| `declared-quantized module shipping BF16 is refused` | D2 cross-check, direction 1 |
| `excluded module shipping NVFP4 is refused` | D2 cross-check, direction 2 |
| `NVFP4 tensors with no declaration are refused` | D2 cross-check, direction 3 |
| `an unimplemented quant_algo is refused by name` | `quant_algo: "FP8"` names the algorithm, not a dtype |
| `a non-ModelOpt quant_method is refused by name` | `quant_method: "awq"` |
| `wildcard exclusions are honoured` | the `fnmatch` pass, which the published artifact does not exercise |
| `the W4A4 notice names both algorithms` | D6 |

## Gates

- Focused: `ctest -R test_qwen3_dflash2_nvfp4` plus the two neighbouring draft
  suites `test_qwen3_dflash2_exl3` and `test_qwen3_dflash2_draft`, which are the
  arms this change must leave byte-unchanged.
- Full: `scripts/agent-preflight.sh --staged`, judged by grepping for
  `gate(s) failed` and `NOT a green`.
- Reachability, by mutation, restored byte-for-byte with a sha256 either side:
  delete the `cfg.raw["quantization_config"]` carry in
  `MakeQwen3DFlashDraftConfig` and confirm the arm cases red. A gate that stays
  green without that line is measuring a class, not a capability.
  **RUN ON THE MERGED TREE: 11 of the 15 cases red under it**, including the
  published-shape load, every refusal that names a module, and both cross-check
  directions; the four that survive are the ones whose subject is the tensors
  rather than the declaration.
  `src/vllm/model_executor/models/qwen3_dflash_weights.cpp` hashed
  `1b5ecb7c6ea1604386caec7442c82f6a4b7b6eb96bd09a6fee069fcb6200e2ec` before the
  mutation and the same value after the restore, and the suite returned to
  15/15 on the restored file. (The pre-merge run of the same mutation read 10 of
  14 at hash `093fa70722f56d8f30744d051235178b77cff2de937b2b9262e897e63c5cba4c`;
  the merge added no case and moved no line in this file, so the two differ only
  by the lane case added after that run.)

## Risks and decisions

- **R1. A wrong `weight_scale_2` is invisible.** An NVFP4 weight read with the
  wrong global scale is correctly distributed and entirely wrong, and the draft
  would still emit the target's tokens. The gate therefore asserts the scalar
  value on every owner rather than only that the owner is non-empty.
- **R2. This row cannot prove the drafter runs.** `vt::MatmulNvfp4` is
  registered for CUDA only (`src/vt/cuda/cuda_matmul_nvfp4.cu`), so no CPU
  forward twin is possible and the gate is a load gate. That is the same
  boundary the target's own NVFP4 arm sits behind, and it is why leg F of
  `BENCH-QWEN38-27B-SOTA` remains a device leg.
- **R3. Acceptance is the axis a token gate cannot see.** Every refusal above
  exists because the alternative is a draft that loads, drafts worse, and shows
  no symptom: the verify is lossless, so only acceptance falls.

## Owed

- **The device leg.** No GPU has loaded this artifact. Leg F of
  `.agents/specs/bench-qwen38-27b-nvfp4-matched.md` is the leg that pays it, and
  it needs a lease. Until it runs, this arm is proven to READ the format and not
  to RUN it.
- ~~**A locally computed sha256 for `model.safetensors`.**~~ **ALREADY PAID, by
  `BENCH-QWEN38-27B-SOTA`.** This row read the header by HTTP range request and
  downloaded nothing, but the payload is already staged and hashed:
  `2228b9b22e93a88d84556419c879448ab6c490ae65c4c0b166f4962190ddbf26`, computed
  from the local bytes and matching the publisher's LFS object hash, recorded in
  `docs/USAGE.md`. This row records no hash it did not compute and quotes that
  one from the row that did.
- **THE FORWARD BRANCH IS NOT REACHED BY ANY GATE IN THIS TREE, and this bullet
  is the declaration `AGENTS.md` §"Nothing lands dead" requires for it.** The
  nine `DflashLinear` / `DflashMlpGateUp` call sites ARE reached and gated --
  every existing bf16 and EXL3 draft suite runs through them, and deleting one
  reds those suites -- but the branch that binds `Nvfp4W4A16LinearMethod` needs
  a populated `Nvfp4Weight`, and executing it needs `vt::MatmulNvfp4`, which is
  registered for CUDA only (`src/vt/cuda/cuda_matmul_nvfp4.cu:2703`). So this
  change lands a LOADER capability that is gated and a FORWARD branch that is
  compiled, routed and unexecuted. It is owned by this row,
  `MODEL-DFLASH2-NVFP4`, tracked by
  [#2758](https://github.com/mudler/vllm.cpp/issues/2758), and the leg that
  executes it is leg F of `BENCH-QWEN38-27B-SOTA`
  ([#2761](https://github.com/mudler/vllm.cpp/issues/2761)).
- **NVFP4 owners for `fc`, the selector projection and the conv kernel
  projections.** Refused by name today. Owed by this row; a published draft that
  quantizes any of them is the trigger, and none does.
- **The compressed-tensors spelling on a draft.** No published DFlash draft
  ships it.
- **The FP8 KV-cache arm**, if a calibrated draft ever ships `k_scale`/`v_scale`:
  owned by `KV-FP8`, [#1593](https://github.com/mudler/vllm.cpp/issues/1593).
- **The W4A4 activation path itself**, which this row makes visible and does not
  fix: owned by `QUANT-QWEN38-27B-NVFP4-ARM`,
  [#2760](https://github.com/mudler/vllm.cpp/issues/2760).
- The `## Outcome` section this spec owes at `DONE`.

## Stop conditions

- The published artifact's header disagrees with the shape above: stop and
  re-read it, do not widen the reader to fit.
- A refusal above would refuse a checkpoint this tree already loads: stop. The
  BF16 and EXL3 drafts must be byte-unchanged, and the inertness case is the
  gate for that.
