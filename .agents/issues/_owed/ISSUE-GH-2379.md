ID: ISSUE-GH-2379
Title: The GPU runner never sets ModelForwardInput.mm, so a Qwen3-VL server throws on the first forward step of every request (re-files hidden #2300)
Row: -
State: OPEN
Kind: UNKNOWN
GitHub: 2379
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
> **Re-files #2300**, which is invisible: it was created by an automation account that has since been suspended, and GitHub hides a suspended account's content rather than deleting it. `#1358`, `#2257`, `#883`, `#2317`, `#2190`, `#2191`, `#2193` and `#2176` are hidden the same way. `main` and several specs cite those numbers, so the references dangle for any reader — that record reconciliation is owed separately and is not this issue.
>
> ## The defect
>
> `src/vllm/v1/worker/gpu/runner.cpp` never sets `ModelForwardInput::mm`. A `grep -cE 'mm_features|MultiModalForwardInput|\.mm = '` over its ~4400 lines returns **0**.
>
> `ForwardQwen3VLForConditionalGeneration` (`src/vllm/model_executor/models/qwen3_vl_registry.cpp`) makes it **mandatory**:
>
> ```cpp
> VT_CHECK(input.mm.has_value(), "Qwen3VLForConditionalGeneration registered forward requires multimodal inputs ...");
> ```
>
> So a server started on a Qwen3-VL checkpoint throws on the first forward step of **any** request, text or image. Gemma-4 (`gemma4_registry.cpp`) and Muse Glimmer (`muse_glimmer_registry.cpp`) instead use `if (input.mm.has_value())` with a text fallback — the mandatory form is Qwen3-VL's alone.
>
> ## Where the data dies
>
> `Request.mm_features` is populated (`src/vllm/v1/engine/input_processor.cpp`) and has one real consumer, the prefix-cache extra keys. It is dropped at `NewRequestData::from_request` (`include/vllm/v1/core/sched/output.h`), which has no mm field. The scheduler has no encoder scheduling at all: `EncoderCacheManager` is fully ported and **constructed nowhere in the tree**. The runner's `CachedRequestState` records `mm_features` as DEFERRED and has no MRoPE fields.
>
> ## Scope of the fix
>
> Three hops, and they should land together — splitting them leaves the middle one unreached and, worse, opens a correctness hole: upstream's encoder-cache-miss error is unreachable only because the scheduler refuses to chunk across an image, and `ServerHarness` runs with chunked prefill enabled.
>
> 1. `NewRequestData.mm_features`; `CachedRequestState.mm_features` plus `mrope_positions` / `mrope_position_delta`; `EncoderCacheManager` constructed in the scheduler; `_try_schedule_encoder_inputs`; `scheduled_encoder_inputs`; `free_encoder_mm_hashes`.
> 2. Runner: the encoder step (which becomes the first `src/` caller of a vision tower forward), gather, merge, MRoPE delta, and setting `.mm`.
> 3. A server-level end-to-end gate through the real `ApiServer`.
>
> ## Evidence it is real
>
> Statically derived from the greps above. Corroborated by a real run: `.agents/specs/multimodal-track.md` records the `qwen3-vl` benchmark arms stopping at `/health` and failing to complete a request on thor:gpu0 (`main` `41ab550b9`, 2026-08-24) because this forward refuses text-only input by name. That observation is the text half; the image half has the same cause.
>
> ## Not blocked on the seam
>
> The forward contract already carries device handles — `MultiModalForwardInput` holds borrowed `vt::Tensor` views as of `b4f0219bb`. What is missing is the runner populating it.
>

## Resolution

-
