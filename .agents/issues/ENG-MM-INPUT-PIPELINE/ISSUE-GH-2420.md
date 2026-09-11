ID: ISSUE-GH-2420
Title: toy_model_plugin's kToyInfo is positional over a ModelInfo whose leading fields are all bool, so a mid-struct insert misassigns SILENTLY
Row: ENG-MM-INPUT-PIPELINE
State: OPEN
Kind: UNKNOWN
GitHub: 2420
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `ENG-MM-INPUT-PIPELINE`
>
> `tests/vllm/plugins/toy_model_plugin.cpp:67` initialises `vllm::ModelInfo kToyInfo`
> positionally:
>
> ```cpp
> constexpr vllm::ModelInfo kToyInfo{
>     /*is_text_generation_model=*/true,
> };
> ```
>
> `ModelInfo` (include/vllm/model_executor/models/model_registry.h:64) opens with five
> consecutive `bool` fields: `is_text_generation_model`, `is_pooling_model`, `is_hybrid`,
> `has_inner_state`, `supports_multimodal`.
>
> Insert any new `bool` ahead of `is_text_generation_model` and that `true` silently becomes
> the new field's value while `is_text_generation_model` falls back to its default `false`.
> It compiles clean. No diagnostic fires.
>
> This is the same defect class that #2398 hit on the sibling struct, and the reason to file
> it separately is that the sibling was survivable and this one is not. There, `ModelFactory`
> gained `encode_mm` mid-struct and `kToyFactory`'s positional initializer put a `bool` into a
> function-pointer slot -- a type error, caught loudly by four CI lanes:
>
> ```
> tests/vllm/plugins/toy_model_plugin.cpp:65:1: error: cannot convert 'bool' to
>   'vllm::ModelEncodeMmFn' ... in initialization
> ```
>
> `ModelInfo`'s leading run of same-typed fields removes exactly that protection. The failure
> would instead be a plugin that quietly reports the wrong capability, and the tests asserting
> `is_text_generation_model` would read the misassigned value as the truth.
>
> #2398 converted `kToyFactory` to designated initialization and enumerated all 37
> `ModelFactory` initialisers in the tree (the other 36 were already designated). `kToyInfo`
> was noted in that work and deliberately left, because changing it there would have grown a
> pull request that was already at review. It is the last positional initialiser of a
> registration struct in the tree.
>
> Fix: convert `kToyInfo` to designated initialisation. One line, no behaviour change.
>
> Worth pairing with a sweep for any other positional initialiser over a struct whose leading
> fields share a type, since that combination is what makes the misassignment silent rather
> than a compile error.
>

## Resolution

-
