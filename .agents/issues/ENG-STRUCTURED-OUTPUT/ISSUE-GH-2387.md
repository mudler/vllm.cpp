ID: ISSUE-GH-2387
Title: Jump-forward decoding ships a server flag, an ABI field and a FEATURES tick, and its capability function has no caller
Row: ENG-STRUCTURED-OUTPUT
State: OPEN
Kind: UNKNOWN
GitHub: 2387
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What
>
> The capability terminates in a bool nobody reads.
>
> ```
> $ git grep -n "DrainForcedTokens" -- src include tests examples
> include/vllm/v1/structured_output/jump_forward.h:70   (decl)
> src/vllm/v1/structured_output/jump_forward.cpp:25     (def)
> tests/vllm/v1/structured_output/test_jump_forward.cpp:39,185,288,294
>
> $ git grep -n "jump_forward_enabled()" -- src include tests
> include/vllm/entrypoints/model_loader.h:581   (the accessor itself)
> tests/capi/test_capi.cpp:1337,1345,1353
> ```
>
> `DrainForcedTokens` has no production caller, and the accessor that would gate it is read only by tests.
>
> ## Everything up to the latch IS wired, which is what makes it convincing
>
> - `src/vllm/entrypoints/openai/server_main.cpp:633-645` parses `--enable-jump-forward` and **hard-errors if passed twice** — a strong "this flag is real" signal
> - `:1293` assigns it
> - `src/capi/vllm_c.cpp:712-727` validates the ABI field
> - `model_loader.cpp:2022-2023` resolves it into `jump_forward_enabled_`
>
> Then nothing consumes it.
>
> ## What the user-facing surfaces promise
>
> - `include/vllm.h:77,551-560` — ABI **v10** added `vllm_model_params.enable_jump_forward` as a public tri-state field
> - `docs/reference/server.md:202` — `| --enable-jump-forward | off | Jump-forward decoding for structured output (token-unique subset) |`
> - `docs/FEATURES.md:278` — `| Jump-forward decoding | ✅ opt-in |`
> - `docs/SGLANG-COMPAT.md:155` shows a copy-pasteable `server --model ... --enable-jump-forward`
>
> `docs/ENVIRONMENT.md:120` is honest — "Currently drives only the standalone driver (`DrainForcedTokens`) ... Off by default until the production scheduler splice ... lands" — but `server.md` and `FEATURES.md` are the two a user reads first, and neither says so.
>
> ## Fix, either direction
>
> Land the scheduler splice, or correct `server.md` and `FEATURES.md` to match `ENVIRONMENT.md`'s honesty. What must not persist is a documented, validated, ABI-versioned flag that provably does nothing.
>
> Row: `ENG-STRUCTURED-OUTPUT`

## Resolution

-
