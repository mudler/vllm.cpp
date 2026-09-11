ID: ISSUE-GH-193
Title: A100 (sm_80) first run: crashes + wrong GDN output in fast paths
Row: BACKEND-CUDA-SM080
State: OPEN
Kind: bug
GitHub: 193
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-09
Updated: 2026-08-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Env: A100-SXM4-80GB · CUDA 13.0 · commit f921062b
> Build: add -DVLLM_CPP_CUDA_ARCHITECTURES=80 + -DCUTLASS_ENABLE_TOOLS=OFF (CUDA 13 otherwise fails in CUTLASS tools/library: duplicate sm_100f flags).
> Result: 342/358 tests pass. 14 stable failures:
> - Crashes (illegal memory access): GDN, MLA, FA2, and model/engine tests — test_qwen3_5_gdn_spec_routing, test_cuda_deepseek_v4, test_mla_attention_block, test_llama_embedding_fold, test_llm_engine, test_openai_api_server, test_minimax_h3
> - Wrong output: GDN projection hash ≠ vLLM oracle (test_op_parity)
> - Expected on sm_80 (GB10/Blackwell-gated): fp8/fp4 capability + sm120a-only op tests
> Impact: GDN/MLA/FA2 models may crash or diverge on A100. Happy to bisect with compute-sanitizer.

## Resolution

-
