ID: ISSUE-GH-2447
Title: No production path gives an EXL3 checkpoint a paged KV cache: DeepseekV4ForwardExl3Paged has only test callers
Row: MODEL-DSV4-EXL3
State: OPEN
Kind: UNKNOWN
GitHub: 2447
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `MODEL-DSV4-EXL3`
>
> `DeepseekV4ForwardExl3Paged` is the only DeepSeek-V4 forward that takes both a
> paged KV cache and a `DeepseekV4CompressorState`. It has NO production caller.
>
> ```
> $ grep -rn "DeepseekV4ForwardExl3Paged" src/ examples/
> src/vllm/model_executor/models/deepseek_v4.cpp:3738   (its own refusal message)
> src/vllm/model_executor/models/deepseek_v4.cpp:3740   (its own refusal message)
> ```
>
> Every other reference is in `tests/vllm/models/test_deepseek_v4_exl3_loader.cpp`.
>
> The registered forward `ForwardDeepseekV4ForCausalLM`
> (`deepseek_v4_registry.cpp:102-133`) has three branches, and none of them reaches
> it:
>
> - `input.gather_logits` -> `DeepseekV4Model::ForwardDevice`, which binds
>   `dev_be.exl3` and neither `paged_kv` nor `compressor`.
> - `input.multi_kv != nullptr` -> `DeepseekV4ForwardGgufPaged`, which refuses by
>   name on `!weights.has_gguf_weights`. An EXL3 checkpoint has none, so this
>   branch is a refusal for this arm rather than a path.
> - otherwise -> `DeepseekV4Model::Forward` -> `DeepseekV4ForwardExl3`, which also
>   binds neither.
>
> So no production path gives an EXL3 checkpoint a paged KV cache. `V4Backend`
> documents what that means: "Null = stateless full-recompute (the default / --gpu
> path)". The EXL3 arm therefore recomputes the entire prefix on every decode step.
>
> WHY THIS MATTERS MORE THAN #2442. The published 44-47 tok/s target is measured at
> 384k context, 1 seq. A forward that recomputes 384k tokens per step is not a
> decode, and no amount of expert-tower residency or speculation changes that.
> #2442 (routed experts on a CPU queue) is real and now fixed, but this sits in
> front of it.
>
> It also means the DSA composition -- the compressor and indexer work that W3
> landed -- is unreachable in production, since `compressor` is only ever non-null
> on the paged entry. Its gates are real; its only callers are tests.
>
> `AGENTS.md` §"Nothing lands dead": a staged slice may land unreached only when
> the commit body, the PR body and the row's `## Owed` all name it. This one is not
> recorded anywhere as unreached.
>
> Found while wiring #2442, by checking whether the device-staging call sites were
> on the hot path.
>

## Resolution

-
