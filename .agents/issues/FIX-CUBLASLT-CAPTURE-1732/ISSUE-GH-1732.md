ID: ISSUE-GH-1732
Title: Failure with default settings (CUDA graph enabled) on SM80
Row: FIX-CUBLASLT-CAPTURE-1732
State: CLOSED
Kind: bug
GitHub: 1732
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-22
Updated: 2026-08-24
Closed: 2026-08-24

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What broke
>
> On a CMP 170HX (GA100, sm_80, 70 SMs, driver 610.57.104, CUDA 13.3 toolkit V13.3.73, main @ 08c81a8), the default decode path dies on the first CUDA graph capture:
>
> ```
> engine-fatal: EngineCore busy loop threw: vt cuda: matmul: btcublasLtMatmulAlgoGetHeuristic: cublas status 14 (CUBLAS_STATUS_INTERNAL_ERROR)
> ```
>
> With `VLLM_CPP_CUDAGRAPH=0`, the same workload runs correctly without crashing and generates text. Warm decode throughput on that eager path measured 24.3 tok/s for Qwen3.8-27B-UD-Q8_K_XL.gguf, three deterministic runs.
>
> ## Reproduce
>
> ```sh
> build-cuda/examples/vllm-cli \
>   --model /home/models/Qwen3.8-27B-UD-Q8_K_XL.gguf \
>   --prompt "The quick brown fox" --max-tokens 8
> # fails as above
>
> VLLM_CPP_CUDAGRAPH=0 build-cuda/examples/vllm-cli \
>   --model /home/models/Qwen3.8-27B-UD-Q8_K_XL.gguf \
>   --prompt "The quick brown fox" --max-tokens 24
> # succeeds
> ```
>
> ## Root cause
>
> `MatmulBtKernelCuda` queries `cublasLtMatmulAlgoGetHeuristic` on every call (`src/vt/cuda/cuda_matmul.cu:346`). On CUDA 13.3 cuBLASLt, that query fails while a created stream is in graph capture. The cuBLASLt trace names the internal cause:
>
> ```
> CUBLASLT_LOG_LEVEL=4 ... [Error][cublasLtMatmulAlgoGetHeuristic] Could not obtain green context information
> ```
>
> Minimal probe, no vllm.cpp code, CUDA 13.3 nvcc, `-arch=sm_80`:
>
> | Call site state | Result |
> |---|---|
> | Eager | SUCCESS |
> | Capture active on a created stream | status 14, INTERNAL_ERROR |
> | Capture active on the legacy null stream | SUCCESS |
> | After capture | SUCCESS |
>
> Prefill runs the row-major NN path eagerly, so the first `matmul_bt` call of a request is also the first call inside the decode capture region. Upstream vLLM does not hit this because torch caches the selected cuBLASLt algo per shape and vLLM runs eager warmup steps before capture.
>
> ## Fix direction
>
> Cache the selected heuristic per shape key outside the capture path, following the existing plan-cache pattern (`fp8_plan_cache.h`), and warm the `bt` path eagerly before the first decode capture.
>
> ## Environment
>
> - GPU: NVIDIA CMP 170HX 64GB (GA100, sm_80, 70 SMs), driver 610.57.04
> - Toolkit: CUDA 13.3 (V13.3.73)
> - Build: `-DVLLM_CPP_CUDA_ARCHITECTURES=80`, FA2 enabled, 0 errors, 1546/1546 targets
> - sm_80 is labeled `build-verified`, not `run-verified`; this board ran it.

## Resolution

GitHub records closing pull request #1741 (https://github.com/mudler/vllm.cpp/pull/1741) merged on 2026-08-24 as commit `3e4cd6d11a2da8a440fcc5e4a36c1d890bbb2cf4`. GitHub closed issue #1732 on 2026-08-24.
