ID: ISSUE-GH-3053
Title: bench(BACKEND-GATE-ROCM-SGLANG): qualify Qwen3-4B BF16 for matched Strix comparison
Row: BACKEND-GATE-ROCM-SGLANG
State: OPEN
Kind: UNKNOWN
GitHub: 3053
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-08
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-GATE-ROCM-SGLANG`
>
> Owner: current Strix campaign operator. User approves Qwen/Qwen3-4B BF16 at 1cfa9a7208912126459214e8b04321603b3df60c for a matched four-engine Strix Halo benchmark: vllm.cpp, pinned vLLM, pinned SGLang, and stock llama.cpp b10451. Spec: .agents/specs/strix-four-engine-qwen3-4b.md (to be committed before implementation).
>
> Build an isolated pinned SGLang ROCm environment for gfx1151; the existing CUDA/aarch64 recipe and MI300/MI350 Docker stages do not qualify. Use native safetensors for ours/vLLM/SGLang and audit a BF16 GGUF conversion for llama.cpp against all checkpoint tensor values and tokenizer IDs. Use six identical raw prompts, 128 output tokens, greedy sampling, BF16 weights and KV, concurrency 1 and 4, warm-up and three measured repetitions, serial engine execution under one Strix lease. Preserve per-request results, resolved settings, hashes, build logs, memory and contention evidence. Correctness precedes acceptance of speed ratios. Record every engine, including unsupported or failing arms; never substitute weights or eager vLLM settings.
>
> Acceptance: all four demonstrably build and run the same model on Strix; exact prompt IDs and output counts; applicable correctness gate; repeated matched throughput/latency/memory with raw evidence; fresh implementation, mutation review, and operator gates. No predetermined speed winner. Links: #3043 and #3048 establish the prior 27B vLLM runtime, not this new model or benchmark.

## Resolution

-
