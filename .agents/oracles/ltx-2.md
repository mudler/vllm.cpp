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
treats as text. All eight are incidental byte matches rather than
registrations. `vllm/model_executor/models/registry.py` exists at that
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
them named `ltx2_*`. Those are three named subsets, not a partition: **97** files
in the tree carry the revision. `.agents/porting-inventory.md` (5 hits) and
`.agents/kernel-matrix.md` (1) carry it as well, from `.agents/` — one directory
above the `.agents/specs/` glob, and outside all three.

Six C++ suites carry the 40-hex literal and FIVE of them assert it:
`test_ltx2_tiling.cpp:88,392-393`,
`test_ltx2_pipeline.cpp:63,199`, `test_ltx2_dfr.cpp:105-106`,
`test_ltx2_vae.cpp:1757,1767` and `test_ltx2_image_cond.cpp:79,343-344`
each compare a revision constant and fail when it is any other revision. The
sixth,
`test_ltx2_lora.cpp:9`, carries the literal in a prose header comment only: it
includes no goldens `.inc`, declares no revision constant, and none of its
`CHECK`s touches one, so it cannot fail on a revision advance. Carrying the
string and asserting it are different things, and a `git grep -l` file list
cannot tell them apart.

**None of those five reaches the `pin` field below, and nothing else does
either.** Each compares a constant emitted into its own goldens against one
hardcoded in the test file; none reads `.agents/oracles/`, and all need a C++
build. Flipping one hex digit of `pin` here leaves `check-oracle-pins.py`,
`check-agent-record.py`, `check-gate-commands.py` and the 185 cases of
`test_check_oracle_pins.py`, `test_agent_record.py` and
`test_check_gate_commands.py` green, because the checker is deliberately
network-free and validates the shape of a 40-hex string rather than its value.

Python does carry the revision, and no gate reads it there either. The five
`scripts/` files above hold it, TWO as live module constants rather than prose —
`gen-ltx2-tiling-goldens.py:59` (`_PIN`, the full 40 hex) and
`gen-ltx2-res2s-goldens.py:60` (`PIN`, the 7-char prefix) — and no gate
executes any of the five: neither `scripts/agent-preflight.sh`, nor anything
under `tests/scripts/`, nor any workflow under `.github/workflows/` names
`gen-ltx2-*` or `probe_ltx2_*`. They are generators a human runs by hand.

The defence is the identity block above — a named clone, a clean
worktree, a stated `HEAD` — and re-derivation by a reader, not a gate.

**Not gateable, because nothing has run the model.** Thirteen tracked scripts
import and execute upstream code, and every one of them runs individual modules
at reduced dimensions on synthetic PRNG weights, or reads constants and
safetensors headers — the generated goldens say so in their own headers
(`tests/vllm/models/ltx2_vae_goldens.inc:3-6`: "Weights and inputs come from the
shared deterministic stream, so no weight byte is checked in").
`ltx2_pipeline_goldens.inc` repeats that sentence verbatim.
`ltx2_goldens.inc` and `ltx2_text_goldens.inc` carry the operative clause inside
a different sentence — "the MATH is gated exactly here and no weight byte is
checked in" — and `ltx2_tiling_goldens.inc:6-8` and
`tests/vllm/multimodal/ltx2_image_cond_goldens.inc:4-6` state it in their own
words. All six say no weight byte is checked in; only two say it in the same
words, and other LTX-2 artifacts carry the clause as well, so these six are a
sample rather than the population.

Of those thirteen, the two that touch real checkpoint bytes run one
`AdaLayerNormSingle` module
(`scripts/measure-ltx2-prompt-adaln.py:126-140`, the forward itself at :140)
and a meta-device loader probe
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
