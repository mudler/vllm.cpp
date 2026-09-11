ID: ISSUE-GH-2408
Title: ENG-MM-INPUT-PIPELINE: the five paths P2 made reachable without making complete
Row: ENG-MM-INPUT-PIPELINE
State: OPEN
Kind: UNKNOWN
GitHub: 2408
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
> P2 (#2379, landed by #2398) made the runner's multimodal path reachable: `GPUModelRunner` runs the registration's `encode_mm` per item and `embed_mm` per step and fills `ModelForwardInput::mm`, so a served Qwen3-VL image request is answered through `ModelRegistry::Forward`. It made five paths REACHABLE without making them COMPLETE, and each one refuses by name in the code rather than producing an answer.
>
> Those five are listed under `## Owed` in `.agents/specs/multimodal-track.md` (§"Owed by P2 — the runner multimodal path"). They pointed at #2379 as their owner, and #2398 CLOSES #2379 — so the owner would have been a closed issue. This issue is the live owner; #2379 stays as the history of how the path landed.
>
> ## What is owed
>
> 1. **Qwen3-VL serves ONE request per step.** `ForwardQwen3VLForConditionalGeneration` returns the last token's logits and does not read `input.logits_indices`, so a batched step cannot be answered; it refuses with `num_reqs <= 1`. Closing it is a per-row gather inside the VL forward (`VLForwardLastLogitsDBuf`), which moves the numbers the M2c golden was measured on and therefore needs its own gate run.
> 2. **The merge pays a HOST round-trip.** `EmbedMmQwen3VLForConditionalGeneration` downloads the gathered tower rows, runs `Qwen3VLMergeMultimodal` and `Qwen3VLComputeDeepstack` on the host in f32, and uploads the result. That is exactly the arithmetic the gated M2c driver runs, which is why it was chosen. A device-resident merge is a measured change against that golden, not a cleanup.
> 3. **Only the `image` modality reaches the runner.** `EncodeMmQwen3VL...` refuses `video` and `audio` by name. Qwen3-VL has a video tower and a video driver but no runner path: the video item's placeholder structure is timestamp-interleaved and needs `Qwen3VLGetRopeIndexVideo`, a different M-RoPE entry point from the one the hook calls.
> 4. **Gemma-4's `ForwardMm` and Muse Glimmer's are still compile-only.** Neither registration declares `encode_mm` / `embed_mm`, so `ModelRegistry::SupportsMmInputs` is false for them and the runner's multimodal arm is never entered. Declaring the hooks is the small half; the blocker is that neither model has a chat seam or a processor producing `mm_features`.
> 5. **The seam install in `server_main.cpp` has no reachability mutation of its own.** `chat.set_multimodal_chat_fn(...)` is the production install and it landed before P2; the P2 e2e gate installs the same seam itself, because a unit test cannot drive `server_main`'s argv path. Deleting the production line leaves the P2 suite green.
>
> ## Where the record lives
>
> `.agents/specs/multimodal-track.md` `## Owed`, block "Owed by P2 — the runner multimodal path". Each of the five entries names this issue as its owner.
>

## Resolution

-
