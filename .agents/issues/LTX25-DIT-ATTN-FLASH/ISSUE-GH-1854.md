ID: ISSUE-GH-1854
Title: **NOT GATEABLE in this tree, declared rather than proxied: nothing asks whether an LTX-2.5 render is GOOD, only whether two renders are the SAME.** Raised while relocating #1743's criterion. The RELATIVE form of the question IS now answered and gated - the coherence checks of §11.3 assert that neither arm is systematically sharper, blockier, quieter or less mobile than the other, at `K <= 0.5` where a one-directional degradation gives `K = 1` exactly. The ABSOLUTE form is not answered: **prompt adherence needs a vision-language model**, which §10.8 already refuses to approximate ("a check for 'is this a golden retriever shaking off water' is a model, not a threshold"), and **artefact-freedom needs an absolute reference render** from an oracle that runs this pipeline, which `.agents/oracles/` does not have. `scripts/ltx25-render-compare.py` therefore computes an **absolute quality panel per arm** - 8-grid and 32-grid blockiness ratios, clipped-pixel fraction, mean sharpness - prints it, records it in the JSON and **checks none of it**, saying so in its own output, rather than inventing a threshold that means nothing without a reference. NOT FIXED IN FLOW and deliberately not: a proxy for perceptual quality is the `a-shape-valid-gate-passes-a-wrong-artefact` failure. Owned by row `LTX25-DIT-ATTN-FLASH` and listed under `## Owed` in §11.5
Row: LTX25-DIT-ATTN-FLASH
State: UNKNOWN
Kind: bug
GitHub: 1854
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:699`

### Frozen archive evidence

> | [#1854](https://github.com/mudler/vllm.cpp/issues/1854) | `LTX25-DIT-ATTN-FLASH` | **NOT GATEABLE in this tree, declared rather than proxied: nothing asks whether an LTX-2.5 render is GOOD, only whether two renders are the SAME.** Raised while relocating #1743's criterion. The RELATIVE form of the question IS now answered and gated - the coherence checks of §11.3 assert that neither arm is systematically sharper, blockier, quieter or less mobile than the other, at `K <= 0.5` where a one-directional degradation gives `K = 1` exactly. The ABSOLUTE form is not answered: **prompt adherence needs a vision-language model**, which §10.8 already refuses to approximate ("a check for 'is this a golden retriever shaking off water' is a model, not a threshold"), and **artefact-freedom needs an absolute reference render** from an oracle that runs this pipeline, which `.agents/oracles/` does not have. `scripts/ltx25-render-compare.py` therefore computes an **absolute quality panel per arm** - 8-grid and 32-grid blockiness ratios, clipped-pixel fraction, mean sharpness - prints it, records it in the JSON and **checks none of it**, saying so in its own output, rather than inventing a threshold that means nothing without a reference. NOT FIXED IN FLOW and deliberately not: a proxy for perceptual quality is the `a-shape-valid-gate-passes-a-wrong-artefact` failure. Owned by row `LTX25-DIT-ATTN-FLASH` and listed under `## Owed` in §11.5 | bug |

## Resolution

-
