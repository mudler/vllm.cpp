ID: ISSUE-GH-629
Title: Metal build fails: unused const kGdnLayers in test_qwen27n_fp8_tower_paged_engine.cpp
Row: BACKEND-METAL-MLX
State: OPEN
Kind: UNKNOWN
GitHub: 629
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-13
Updated: 2026-08-13
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> `tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp:95` declares `constexpr uint64_t kGdnLayers = 48;` unconditionally, but its only uses (lines 281, 282, 329) sit inside `#ifdef VLLM_CPP_CUDA`.
>
> On a non-CUDA build (`cmake -DVLLM_CPP_MLX=ON` Metal configure on macOS) this trips `-Werror,-Wunused-const-variable` and breaks the build at 100%:
>
> ```
> tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp:95:20: error: unused variable 'kGdnLayers' [-Werror,-Wunused-const-variable]
> ```
>
> Fix: move the constant (and its comment) inside the same `#ifdef VLLM_CPP_CUDA` guard as its uses.

## Resolution

-
