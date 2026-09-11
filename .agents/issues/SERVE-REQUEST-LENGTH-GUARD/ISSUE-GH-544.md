ID: ISSUE-GH-544
Title: serve: VT_SERVER_MAX_NEW_TOKENS does not cap unset or negative max_tokens
Row: SERVE-REQUEST-LENGTH-GUARD
State: OPEN
Kind: UNKNOWN
GitHub: 544
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-12
Updated: 2026-08-12
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Problem
>
> The OpenAI serving layer advertises `VT_SERVER_MAX_NEW_TOKENS` as an operator ceiling, but both chat and completion handlers call `request.to_sampling_params()` without passing the ceiling as the serving default. Protocol normalization correctly turns non-positive client limits (including Hermes `max_tokens=-1`) into `std::nullopt`; the subsequent serving clamp only reduces an already-positive value. The unset case therefore bypasses the configured ceiling and expands to `max_model_len - prompt_len` in `InputProcessor`.
>
> ## Reproduction
>
> Live Gemma-4 agent request on `:8010`:
>
> - environment: `VT_SERVER_MAX_NEW_TOKENS=4096`
> - wire request: `max_tokens=-1`
> - provider request log: `max_tokens=-1`
> - a stochastic tool-emission failure generated more than 2,250 tokens without EOS before the client moved on; the configured ceiling was never applied at serving admission.
>
> Source chain on current main:
>
> 1. `ChatCompletionRequest::to_sampling_params()` maps `<=0` to the optional serving default.
> 2. `OpenAIServingChat` and `OpenAIServingCompletion` call it with no default.
> 3. Chat's local clamp uses `value_or(cap)` only for comparison, but does not assign the cap when the optional is unset.
> 4. Completion has no equivalent operator-cap application.
>
> ## Expected behavior
>
> When `VT_SERVER_MAX_NEW_TOKENS > 0`, it is a hard serving ceiling for all requests:
>
> - unset/non-positive client max -> cap;
> - positive client max above cap -> cap;
> - positive client max below cap -> unchanged;
> - cap `0` -> disabled, preserving the protocol's unbounded-to-context behavior.
>
> Apply the policy consistently to `/v1/chat/completions` and `/v1/completions`. Keep protocol normalization unchanged.
>
> ## Acceptance
>
> - RED-first shared-policy unit test covers all four cases above.
> - Both chat and completion handlers use the shared policy.
> - Existing protocol tests proving non-positive values are UNSET remain unchanged and green.
> - Focused OpenAI protocol/serving/API tests pass.
> - Full CPU gate status is reported honestly; unrelated baseline failures are not weakened.
>

## Resolution

-
