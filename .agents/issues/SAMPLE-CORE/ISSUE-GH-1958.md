ID: ISSUE-GH-1958
Title: SAMPLE-CORE: DeviceScratch unified-memory 0-copy is a use-after-free (CUDA illegal memory access under min_tokens)
Row: SAMPLE-CORE
State: OPEN
Kind: UNKNOWN
GitHub: 1958
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-26
Updated: 2026-08-26
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Symptom
>
> On a GB10 (unified memory), a streaming request with `min_tokens > 0` crashes the engine:
>
> ```
> engine-fatal: EngineCore busy loop threw: vt cuda: cudaStreamSynchronize: an illegal memory access was encountered
> ```
>
> compute-sanitizer pins it to:
>
> ```
> Invalid __global__ write of size 4 bytes
>   at vt::cuda::ApplyTokenMaskKernel(float*, const int*, const int*, long, long)
>   Host: vllm::v1::apply_min_tokens -> Sampler::forward -> GPUModelRunner::sample_tokens
> ```
>
> The write index is `rows[k] * vocab + cols[k]`, and `rows[k]` reads back as garbage (~1.3e9), so the write is out of bounds.
>
> ## Root cause
>
> `DeviceScratch` (include/vllm/v1/sample/device_scratch.h) on a unified-memory
> backend wraps the caller's host buffer in place (0-copy):
>
> ```cpp
> if (backend_->UnifiedMemory()) {
>   tensor_ = vt::Tensor::Contiguous(const_cast<void*>(host), dtype, device, shape);
> }
> ```
>
> `apply_min_tokens` passes function-local `std::vector<int32_t>` `rows`/`cols`,
> `vt::ApplyTokenMask` launches the CUDA kernel asynchronously on the queue, and
> the vectors are destroyed when `apply_min_tokens` returns — before the kernel
> reads them. The freed host memory is later reused, so the kernel reads garbage
> and writes `logits` out of bounds.
>
> Short prompts / short output often miss the race; `min_tokens + a longer decode`
> makes it reproducible. This affects every `DeviceScratch` consumer that hands it
> a temporary host buffer (min_tokens, logit_bias, min_p, ...).
>
> ## Proposed fix
>
> Make `DeviceScratch` own its bytes on every backend: allocate and copy instead
> of 0-copy wrapping the caller's temporary buffer. These are small derived
> tensors (rows/cols/biases/min_p), so the copy is negligible.
>
> ## Owning row
>
> `SAMPLE-CORE` (spec: .agents/specs/sampling-controls-c7.md)

## Resolution

-
