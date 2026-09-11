ID: ISSUE-GH-2411
Title: Add DeepSeek V4 Flash Vision support
Row: MODEL-MM
State: OPEN
Kind: UNKNOWN
GitHub: 2411
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`
>
> DeepSeek published `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at revision `86f746b36186f0e567729a5c06a8c918caba82a9`. The checkpoint reuses the DeepSeek-V4 text backbone and adds a 32-layer ViT, aligner, image sentinel embeddings, image-span attention visibility, interleaved image prompt encoding, and a DSpark tail.
>
> vLLM at the repository parity pin has no vision path for `DeepseekV4ForCausalLM`; current vLLM, vLLM-Omni, SGLang, and Transformers also expose no complete implementation of this vision variant. The checkpoint repository's minimal PyTorch inference is the only complete executable reference and needs a model-specific oracle pin before it can gate a port.
>
> Scope: add the model and oracle to the roadmap, commit the implementation spec before code, then implement the processor, vision tower and aligner, weight and quantized arms, DeepSeek image merge and visibility semantics, registered multimodal forward, OpenAI multi-image serving, and correctness/performance gates. The spec and implementation use one pull request, with the spec commit first.
>
> The issue closes when the production entry point accepts the pinned checkpoint family through the documented quantized vehicle, ordinary autoregressive image generation is oracle-gated, text-only DeepSeek-V4 stays byte-identical, and the shipped checkpoint details are documented.

## Resolution

-
