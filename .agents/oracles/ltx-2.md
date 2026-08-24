# Lightricks `LTX-2` — the LTX-2.5 model author's own runtime

A **third** repository, neither `vllm-project/vllm` nor `vllm-project/vllm-omni`,
and the one every `file:line` anchor in the LTX-2.5 lane actually resolves in.
Two packages carry it: `packages/ltx-core/src/ltx_core/` is the architecture —
the joint video+audio DiT, the Conv video VAE, the audio VAE and vocoder, the
Gemma text-encoder plumbing, the guiders and schedulers — and
`packages/ltx-pipelines/src/ltx_pipelines/` is the recipe layer: sigma schedules,
samplers, generation-version parameter inheritance and the shipped defaults.

**Prefer vLLM-Omni wherever it implements the pipeline.** It registers `ltx2`
(`vllm_omni/diffusion/registry.py:69-87` at `a4ea67a2`) and its structure is the
one this project mirrors. Reach here where it does not: its `_PIPELINE_RECIPES`
stop at generation 2.3 (`vllm_omni/diffusion/models/ltx2/ltx2_recipes.py:161-166`),
2.5 is open upstream at
[vllm-omni#6066](https://github.com/vllm-project/vllm-omni/issues/6066), and the
2.4/2.5 rows this port adds take their values from here. Record which of the two a
stage was gated against, because they do not always agree — the default negative
prompt differs, Lightricks' carrying five leading tags vLLM-Omni's lacks, and
this port keeps both strings and gates the disagreement as a value
(`kLtx2NegativePromptsAgree`) rather than resolving it by preference.

`diffusers` overlaps here too, and the tie-break is written down rather than left
to whoever reaches for one first. [#1012](https://github.com/mudler/vllm.cpp/issues/1012)
records that `diffusers` at its recorded pin `c6da9936` implements LTX-2.5 in both
decode arms, and that record is `gateable = yes` while this one is not. Reach for
`diffusers` for a *diffusers-shaped* question — a scheduler or VAE as that library
composes it. Reach here for what the model author's own runtime defines, which is
what every LTX-2.5 correctness gate in this tree already runs against. Where they
disagree, this file is the architecture reference and `diffusers` is not; where a
stage was gated, name which one it was gated against.

vLLM proper registers nothing LTX at the parity pin `5559679229`. Searched in
that checkout at that revision: `git grep -inE "\bltx" -- '*.py' '*.md' '*.yaml'
'*.yml' '*.txt' '*.json'` returns no line, `git grep -ilE "lightricks"` returns
no file, and a whole-tree case-insensitive `ltx` search matches eight paths:
seven PNG/SVG documentation assets and the vendored minified
`vllm/entrypoints/serve/instrumentator/static/swagger-ui-bundle.js`, which git
treats as text. All eight are incidental byte matches rather than registrations. `vllm/model_executor/models/registry.py` exists at that
revision and is inside the searched set. So this is a secondary oracle under the
rule that admits one, not a preference.

**Identity, asserted rather than assumed.** The revision below was read from a
clone whose `origin` is `https://github.com/Lightricks/LTX-2.git`, whose worktree
is clean, and whose `HEAD` is `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` — the
merge of `Lightricks/LTX-2#273`, authored 2026-08-11. The repository carries **no
tags**, so `pin_label` uses the package version both `ltx-core` and
`ltx-pipelines` declare at that revision. Nothing here is a claim about what
upstream `main` holds today: this record, like the checker that reads it, is
network-free.

The pin is not new to the tree, only to the registry. `git grep -l fd4ded7f`,
measured at this head rather than quoted, returns **30** files matching
`.agents/specs/ltx25-*.md`, **5** under `scripts/` — four `gen-ltx2-*.py`
generators and `probe_ltx2_tiling_layout.py` — and **15** under `src/`, twelve of
them named `ltx2_*`. `.agents/porting-inventory.md` and `.agents/kernel-matrix.md`
carry it as well, one directory above every glob above. It is already asserted
executably in one place —
`tests/vllm/models/test_ltx2_tiling.cpp:88,392-393` fails when the emitted
`kLtx2TilingUpstreamRevision` is any other revision.

**Not gateable, because nothing has run the model.** Thirteen tracked scripts
import and execute upstream code, and every one of them runs individual modules
at reduced dimensions on synthetic PRNG weights, or reads constants and
safetensors headers — the generated goldens say so in their own headers
(`tests/vllm/models/ltx2_vae_goldens.inc:3-6`: "Weights and inputs come from the
shared deterministic stream, so no weight byte is checked in", a sentence
`ltx2_goldens.inc`, `ltx2_pipeline_goldens.inc` and `ltx2_text_goldens.inc`
repeat verbatim, and `ltx2_tiling_goldens.inc:6-8` and
`tests/vllm/multimodal/ltx2_image_cond_goldens.inc:4-6` state in their own
words). Of those thirteen, the two that touch real checkpoint bytes run one
`AdaLayerNormSingle` module
(`scripts/measure-ltx2-prompt-adaln.py:124-133`) and a meta-device loader probe
(`scripts/measure-ltx2-keyframes-meta.py:156,203`); neither writes a committed
artifact. There is no `tools/oracle/` LTX script and no `tests/parity/goldens/ltx*`
manifest. [#1864](https://github.com/mudler/vllm.cpp/issues/1864) owes the
measurement, and `.agents/specs/oracle-ltx-2-pin.md` §"Owed" lists it.

One consequence worth stating, because it is the reason the run matters rather
than a formality: where the revision is asserted no weight is loaded, and where
weights are loaded only the interpreter *path* is asserted, never the revision.
`.agents/specs/ltx-2-5.md` §7.0(b) records why that is not hypothetical — a decoy
`ltx_core` once produced byte-identical goldens and exited 0.

```oracle-pin
id = ltx-2
role = secondary
upstream = https://github.com/Lightricks/LTX-2
scope = the LTX-2.5 architecture and pipeline recipes from the model author's own runtime, for the generations and defaults vLLM-Omni's ltx2 registration does not reach
pin = fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca
pin_label = ltx-core / ltx-pipelines 1.2.0 (main, merge of LTX-2#273)
pinned_on = 2026-08-24
gateable = no
evidence = #1864
```
