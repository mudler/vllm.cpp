ID: ISSUE-GH-2372
Title: main tip does not build: the DSA schedule test predates W9's signature, and glm5_next_weights trips MSVC -Werror
Row: -
State: CLOSED
Kind: UNKNOWN
GitHub: 2372
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Observed
>
> Against main `da2291339` (first executed on PR #2370's merge ref; main's own run had these jobs cancelled before executing, so the break was invisible there):
>
> - `build-test-cpu` and both `sanitize-cpu` jobs fail compiling
>   `tests/vllm/models/test_glm_moe_dsa_schedule.cpp:304`: `cannot convert 'vllm::mla::MlaSharedSelection*' to 'vt::Tensor*'`.
> - Both `windows-msvc-*` jobs fail at
>   `src/vllm/model_executor/models/glm5_next_weights.cpp(223,31)`: `error C2220` (warning treated as error).
>
> ## Attribution
>
> - The schedule test last changed in `ee5c86031` (MODEL-TEXT-GLM-MOE-DSA W4); `11f34effb` (the same row's W9, "pass W9's shared selection to the parameter it names") changed the signature without carrying the test. **Owner: MODEL-TEXT-GLM-MOE-DSA.**
> - `glm5_next_weights.cpp(223)` last changed in `a36add6a8` (MODEL-MM-GLM53-FLASH #2278). The in-flight `row/BUILD-CPU-WERROR-*` branches guard MXFP4 tests under `VT_MARLIN_NVFP4` and do not touch this file. **Owner: MODEL-MM-GLM53-FLASH.**
>
> Both facets block green CI for every open PR until repaired; neither is caused by any open PR's changes.
>
> FOLLOWING_AGENTS_PROTOCOL
>
> Following-Agents-Protocol: true
> AI-Assisted: true
> Assisted-by: AGENT:zai-glm-5.3-flash [maki]

## Resolution

The binding comment https://github.com/mudler/vllm.cpp/issues/2372#issuecomment-5483226920 dated 2026-08-31 verifies both compile facets on main: commit `08fa2f5aa` repairs the schedule call and commit `5f230020f` contains the KdaHeadCount rename.
