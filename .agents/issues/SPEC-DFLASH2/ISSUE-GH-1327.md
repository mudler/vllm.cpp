ID: ISSUE-GH-1327
Title: `.agents/specs/dflash2-spec-decode.md` `## Upstream chain` said the three output scalars `input_embedding_scale`, `output_multiplier` and `final_logit_softcapping` are "ABSENT from this config" and that "no published checkpoint exercises them, so the port implements them and gates them synthetically". That was measured on `z-lab/Qwen3.8-27B-DFlash2` alone. `z-lab/Muse-Glimmer-30B-DFlash2` — the SECOND published DFlash2 checkpoint, `config.json` sha256 `cb684d6f688a22619a63ea1debe7d30c139c195bf3141fd86a763763ab34b5d9`, read 2026-08-19 — sets `output_multiplier` to `0.19611613513818404` and `final_logit_softcapping` to `20.0`, and ships `block_size` 16 against the 27B's 8, hidden 6656 (416 groups, a 1664-wide `kernel_projection`) and `rope_theta` 500000.0. Both scalars are applied to the candidate VALUES in `compute_candidates` BEFORE the selector scores them, so a wrong value reorders the top-K and moves acceptance without raising — the `is_causal` failure class one layer up, and the class no token gate here can see. A port reading all three with `.get(key, default)` would pass every gate built from the 27B draft and be measuring the default path. The same file also falsifies `## Scope`'s exclusion of "a second DFlash2 target family": upstream registers ONE architecture and both checkpoints declare `model_type` `qwen3`, so what the second adds is values rather than a class. FIXED IN FLOW by `SPEC-DFLASH2` W2, which is the wave that had to read both configs anyway: `## Scope` drops the exclusion, `## Upstream chain` records both values with their source, `## Gates` G1 now requires BOTH published block shapes (upstream's own reference test parametrises 5 and 8 and never reaches 16), and `## Risks/decisions` D9 binds W3 to gate the scalars against the checkpoint that sets them
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: gap
GitHub: 1327
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:484`

### Frozen archive evidence

> | [#1327](https://github.com/mudler/vllm.cpp/issues/1327) | `SPEC-DFLASH2` | `.agents/specs/dflash2-spec-decode.md` `## Upstream chain` said the three output scalars `input_embedding_scale`, `output_multiplier` and `final_logit_softcapping` are "ABSENT from this config" and that "no published checkpoint exercises them, so the port implements them and gates them synthetically". That was measured on `z-lab/Qwen3.8-27B-DFlash2` alone. `z-lab/Muse-Glimmer-30B-DFlash2` — the SECOND published DFlash2 checkpoint, `config.json` sha256 `cb684d6f688a22619a63ea1debe7d30c139c195bf3141fd86a763763ab34b5d9`, read 2026-08-19 — sets `output_multiplier` to `0.19611613513818404` and `final_logit_softcapping` to `20.0`, and ships `block_size` 16 against the 27B's 8, hidden 6656 (416 groups, a 1664-wide `kernel_projection`) and `rope_theta` 500000.0. Both scalars are applied to the candidate VALUES in `compute_candidates` BEFORE the selector scores them, so a wrong value reorders the top-K and moves acceptance without raising — the `is_causal` failure class one layer up, and the class no token gate here can see. A port reading all three with `.get(key, default)` would pass every gate built from the 27B draft and be measuring the default path. The same file also falsifies `## Scope`'s exclusion of "a second DFlash2 target family": upstream registers ONE architecture and both checkpoints declare `model_type` `qwen3`, so what the second adds is values rather than a class. FIXED IN FLOW by `SPEC-DFLASH2` W2, which is the wave that had to read both configs anyway: `## Scope` drops the exclusion, `## Upstream chain` records both values with their source, `## Gates` G1 now requires BOTH published block shapes (upstream's own reference test parametrises 5 and 8 and never reaches 16), and `## Risks/decisions` D9 binds W3 to gate the scalars against the checkpoint that sets them | gap |

## Resolution

-
