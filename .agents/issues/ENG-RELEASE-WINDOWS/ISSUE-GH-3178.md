ID: ISSUE-GH-3178
Title: test_model_loader_gguf: CHECK_THROWS_WITH_AS expected string missing DeepseekV41ForCausalLM
Row: ENG-RELEASE-WINDOWS
State: CLOSED
Kind: UNKNOWN
GitHub: 3178
Mirror: SYNCED
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: 2026-09-13

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `ENG-RELEASE-WINDOWS`
>
> Commit 4a9d33792 added `DeepseekV41ForCausalLM` to the model registry but did not update the hardcoded expected architecture list in `tests/vllm/test_model_loader_gguf.cpp:225`. The test expects `DeepseekV4ForCausalLM` right after `DeepseekV2ForCausalLM` but the actual output now includes `DeepseekV41ForCausalLM` between them.
>
> This causes `build-test-cpu` to fail on main.
>
> ## Fix
>
> Add `'DeepseekV41ForCausalLM', ` to the expected string after `'DeepseekV2ForCausalLM',` in the CHECK_THROWS_WITH_AS call.

## Resolution

fixed by adding DeepseekV41ForCausalLM to the expected string in test_model_loader_gguf.cpp (PR #3179)
