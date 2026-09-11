ID: ISSUE-GH-2403
Title: windows focused gate: test_openai_api_server aborts 0xC0000409 in the explicit-cpu/embeddings cases once the build compiles again
Row: ENG-RELEASE-WINDOWS
State: OPEN
Kind: UNKNOWN
GitHub: 2403
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Observed
>
> PR #2399 unblocked the Windows compile (the C4456/C2220 build break of #2372). The `windows-msvc-cpu` and `windows-msvc-vulkan` focused gates then ran for the first time since the build broke, and `test_openai_api_server.exe` aborts at runtime — no compiler diagnostics, 76/79 cases run, then:
>
> ```
> FAILING CASE 56/79 rc=-1073740791 : api_server: an explicit-cpu device-selected engine serves /v1/completions
> FAILING CASE 57/79 rc=-1073740791 : api_server: embeddings dispatch — OpenAI shape over the engine path
> FAILING CASE 59/79 rc=-1073740791 : api_server: embeddings socket smoke; generate routes 404 on the embedding server
> ```
>
> `rc=-1073740791` is 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN) — the MSVC fast-fail surface for abort()/fail-fast. Job 99460762370 (windows-msvc-cpu), head `9215260148` (#2399), failing step "Build and execute the native Windows CPU focused gate"; the shared `windows-msvc-vulkan` red shows the same class.
>
> ## Attribution
>
> Not caused by #2399: the crashing binary carries neither of that PR's changes' behavior — one is a call-site argument in `test_glm_moe_dsa_schedule.cpp` (a different binary), the other a declaration rename in `glm5_next_weights.cpp` with no behavior change. The defect was unreachable on main while the compile was broken, and main's own windows jobs have been skipped/cancelled across recent heads, so this runtime break has never had a passing build to run in until now — the same first-execution exposure as #2372's build facets.
>
> The failing cases cluster on the explicit-cpu device selection and the embeddings dispatch/socket paths of the api server.
>
> ## Owed
>
> Owner: the api_server row (OPENAI-API-SERVER surface) for the abort, with the windows-gate owner CC'd for the focused-gate harness. The 0xC0000409 needs a Windows reproduction with a debugger or fail-fast hook; this host has no MSVC, so the repair rides CI evidence like the build fix did. Until repaired, both `windows-msvc-*` jobs stay red independently of #2372/#2399.
>
> Found while reading CI for #2399; filed rather than fixed because the abort is outside that row's spec scope (compile breaks) and this host cannot run the failing binary.
>
> FOLLOWING_AGENTS_PROTOCOL
>
> Following-Agents-Protocol: true
> AI-Assisted: true
> Assisted-by: AGENT:zai-glm-5.3-flash [maki]

## Resolution

-
