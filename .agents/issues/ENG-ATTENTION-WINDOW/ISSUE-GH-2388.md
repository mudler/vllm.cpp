ID: ISSUE-GH-2388
Title: The generic attention-window layer is unreachable, and seven models hand-roll the mapping it owns
Row: ENG-ATTENTION-WINDOW
State: OPEN
Kind: UNKNOWN
GitHub: 2388
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
> ```
> $ git grep -c "ResolveAttentionWindow" -- src include tests
> include/vllm/model_executor/layers/attention/attention.h:2   (decl + comment)
> src/vllm/model_executor/layers/attention/attention.cpp:1     (def)
> tests/vllm/model_executor/layers/attention/test_attention.cpp:11
> ```
>
> Eleven references, all in one test file. Its only consumer, `MakePagedAttentionArgs` (`attention.h:33` / `attention.cpp:33`), is likewise declaration, definition and two tests.
>
> The field the pair exists to populate is never written: `v1::AttentionLayer::window_size` (`include/vllm/v1/attention/backend.h:232`) has zero writers. The one production construction, `src/vllm/model_executor/layers/attention/mla_attention.cpp:1080`, is `v1::AttentionLayer layer{}` — value-initialized to `nullopt`.
>
> ## What is bypassed
>
> Every model that needs a sliding window hand-rolls the mapping inline as `vt::AttentionWindow{sliding_window - 1, 0}`: `gemma2.cpp:198`, `gemma3.cpp:193`, `gemma4.cpp:336`, `olmo2.cpp:196`, `muse_glimmer.cpp:230`, `v1/attention/backend.cpp:323`, `mla_chunked_context.h:363`.
>
> The resolver's contract (`attention.h:20-23`) claims it resolves upstream `Attention.__init__` precedence, the model-level disable flag, and FlashAttentionImpl's decoder/encoder mapping, including the encoder-only symmetric-window case and `[1, INT32_MAX]` validation. **None of that runs in production.** The hand-rolled call sites implement only the decoder case and skip the bounds check, so seven copies of a partial rule are live while the complete one is dead.
>
> `docs/FEATURES.md:54` carries `| Sliding-window and chunked-local attention | ◐ | ✅ | ✅ | ✅ |`.
>
> ## Doubly unreachable
>
> The resolver takes a `disable_model_sliding_window` parameter, and no such flag or config key exists anywhere:
>
> ```
> $ git grep -n "disable_sliding_window\|disable-sliding-window" -- src include docs
> (empty)
> ```
>
> ## Note on severity
>
> `attention.h` does say "*Future* model ports set `layer.window_size` with `ResolveAttentionWindow`", which is honest about intent, so this ranks below the flatly-wrong surfaces. It is filed because the shared seam rule says a capability routes through one place: seven inline copies of a narrower rule, plus a complete unreachable one, is the shape that rule exists to prevent.
>
> Row: `ENG-ATTENTION-WINDOW`

## Resolution

-
