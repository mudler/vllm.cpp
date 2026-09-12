ID: ISSUE-GH-3112
Title: fix(BACKEND-ROCM-ATTN-GATE-SPLIT): identify the actual F16 head in the forward test
Row: BACKEND-ROCM-ATTN-GATE-SPLIT
State: OPEN
Kind: UNKNOWN
GitHub: 3112
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM-ATTN-GATE-SPLIT`
>
> The F16 production forward observer identifies an LM-head multiplication solely by its output width in `tests/vllm/test_gguf_keep_quant.cpp` (`F16ForwardObservation::Observe`). The unchanged gated fixture has four other projection calls with that width. After the missing ROCm AttnGateSplit provider and zero rotary-width handling are repaired, both tied and untied cases report five marked heads against the exact-one assertion.
>
> The operator ran the complete frozen HIP target. It reports 58 cases, 57 passing cases, 10,934 passing assertions, and exactly two head-count failures. Retained and expanded F16 logits match and are finite. The frozen evidence is `/home/vikash/.cache/rocm-attn-gate-split-impl/green-focused/gguf-production-full-operator.log`, SHA256 `cf9cd125bce34008bba2bfd0e51f6fbd018facbc552c31e21a5f0f57fdc89d1a`. The command and sealed inputs are retained beside it.
>
> Replace the width-only witness with an exact production head witness. Keep the exact-one assertion, all current model and fixture bytes, both tied modes, default native providers, and retained-versus-expanded numerical comparison. Prove that removing the actual head observation or F16 head marker fails the gate. Do not increase the expected count, loosen the assertion, or alter vocabulary dimensions to hide the collision.
>
> The owning child spec is `.agents/specs/rocm-attn-gate-split.md`. Amend and commit the scoped test design before implementing this in-flow repair. Parent F16 issue #3092 and split issue #3106 remain separately owned. This issue closes with the reviewed change that repairs this test witness.
>

## Resolution

-
