# SAMPLER-HOST-LOGITS-DEVICE-TAG — a host pointer stamped with the CUDA device

Issue: [#1313](https://github.com/mudler/vllm.cpp/issues/1313)
Row: SAMPLER-HOST-LOGITS-DEVICE-TAG
State: ACTIVE

## Scope

`src/vllm/v1/worker/gpu/runner.cpp` builds the `[rows, vocab]` logits tensor the
on-device sampler runs on from a host `std::vector<float>` pointer, and stamps
`queue_.device` on it. On a unified-memory device the address is valid. On a
discrete GPU it is a host address handed to a kernel.

In scope: every site in `runner.cpp` that views a host logits buffer as a tensor
on the queue device, and the shared staging seam they route through.

Out of scope: removing the host-logits path itself (A2-Q2b for `nemotron_h`, the
framework-routing rows for `laguna` and `qwen3_vl`). This row makes the existing
path correct on every backend; it does not change which models take it.

## The defect, grounded

`vt::GreedyArgmax` does NOT copy its input before touching it. The CUDA arm
passes the caller's pointer straight to a kernel:

* `src/vt/cuda/cuda_sample.cu::GreedyArgmaxKernelSlow` — `GreedyArgmaxKernelSlow<<<...>>>(..., logits.Ptr<float>(), v)`
* `src/vt/cuda/cuda_sample.cu::ArgmaxPartialKernel` — `ArgmaxPartialKernel<<<...>>>(..., logits.Ptr<float>(), v, bpr)`

So the pointer IS dereferenced on device. This is a latent illegal access on a
discrete GPU, not a mislabelling.

The same is true of the rest of the sampling pipeline the tensor feeds
(`ApplyTemperature`, `ApplyTopKTopP`, `ComputeProbs`, `ComputeLogprobs` —
`src/vt/ops.cpp::CheckSamplingLogits`), and of `apply_grammar_bitmask`.

## Affected sites (four, not one)

| Site | Path | Host buffer |
|---|---|---|
| `src/vllm/v1/worker/gpu/runner.cpp::assemble_sample_logits` | (A') `VT_GPU_SAMPLE=0` download-then-sample A/B | `sampled_logits` |
| `src/vllm/v1/worker/gpu/runner.cpp::assemble_sample_logits` | host logits, rows already gathered (the reported one) | `fl.host` |
| `src/vllm/v1/worker/gpu/runner.cpp::assemble_sample_logits` | (B) `VT_LOGITS_GATHER=0` host re-gather | `sampled_logits` |
| `src/vllm/v1/worker/gpu/runner.cpp::collect_prompt_logprobs` | `collect_prompt_logprobs` prompt-row slice | `fl.host + offset` |

Affected models are the three that return `ForwardLogits.host`, i.e. exactly the
`scripts/runner-routing-allowlist.txt` entries: `nemotron_h`, `laguna`,
`qwen3_vl`.

## Why it has not bitten

`vt::CudaBackend::UnifiedMemory()` is `caps.pageable_memory_access &&
caps.integrated` (`src/vt/cuda/cuda_backend.cu::UnifiedMemory`). GB10 is integrated, so the
driver services an ordinary host pointer through ATS and the NemotronH A3 gate
reads `96/96 STRICT PASS` at host-memory latency. A discrete GPU reports
`integrated == false` and the same address is illegal.

## Design

The repository already states this contract, in
`include/vllm/v1/sample/device_scratch.h`:

> unified-memory backends (CPU, GB10) wrap the host buffer in place (0-copy);
> discrete backends alloc device memory and copy the host buffer up

`DeviceScratch` applies it to the small derived sampling tensors. The logits
tensor — the largest one, and the one every sampling kernel reads — bypasses it.

Add `HostBufferStaging` beside `DeviceScratch`: the same residency contract, with
one grow-only device allocation reused across steps instead of an alloc/free per
construction, because this sits on the per-token decode path. Route all four
sites through it.

Two staging members, not one: `collect_prompt_logprobs` runs while the assembled
sample-logits tensor is still live (`src/vllm/v1/worker/gpu/runner.cpp::sample_tokens`), so one shared buffer
would invalidate it.

### Alternatives rejected

**The sampler refuses a host pointer carrying a device tag.** Rejected. It
converts a repairable path into a refusal: the three models would stop decoding
on a discrete GPU rather than start working. It also cannot tell a GB10-valid
host pointer from a discrete-invalid one without a per-backend residency probe,
which is the same new seam the repair needs anyway — so it costs what the repair
costs and delivers less.

**Build the tensor with the host device and dispatch accordingly.** Rejected.
`CheckSamplingLogits` (`src/vt/ops.cpp::CheckSamplingLogits`) requires `logits.device ==
q.device` for every sampling op, so a host-device logits tensor needs a host
queue and the whole pipeline re-dispatched to CPU. That regresses GB10 from
on-device sampling to host sampling, and `apply_grammar_bitmask`, the random
path, `compute_prompt_logprobs` and `sample_tokens_async`'s device id buffer all
read the device off the same tensor.

**A residency assertion in `vt::Tensor::Contiguous`.** Rejected as stated in the
issue, and this is worth recording because it looks free. Two registered
backends legitimately stamp a host-dereferenceable pointer with a non-unified
device: Tenstorrent, whose `Alloc` returns `aligned_alloc` host memory by design
(`src/vt/tenstorrent/tenstorrent_backend.cpp::Alloc`, with
`src/vt/tenstorrent/tenstorrent_backend.cpp::UnifiedMemory` returning false), and a discrete Vulkan device, whose buffers are HOST_VISIBLE and
persistently mapped (`include/vt/backend.h::DeviceMemoryIsHostAddressable`). Both register a
`kGreedyArgmax` provider. A blanket assert would false-fire on both. Making it
not false-fire needs a new per-backend residency virtual that every backend has
to get right, on the hottest constructor in the tree, for a debug-only net —
against a repair that removes the defect outright on the four sites that have it.

## Tests

`tests/vllm/v1/sample/test_host_buffer_staging.cpp`, using the fake-backend idiom
already established in `tests/vt/test_reference_tier.cpp` (a `Backend` over host
memory registered on the otherwise-unused `kXPU` slot, one unified instance and
one discrete instance). No GPU required.

1. DISCRETE — the staged tensor's `data` is NOT the host pointer, and it carries
   the host bytes. This is the red.
2. UNIFIED — the staged tensor's `data` IS the host pointer, byte-for-byte the
   zero-copy wrap the GB10 path has today. This makes the "A3 stays green"
   claim executable rather than asserted.
3. Grow-only reuse — a second smaller stage reuses the same allocation; a larger
   one grows it. No per-step alloc churn on the decode path.
4. Two independent staging buffers do not alias, which is what
   `collect_prompt_logprobs` overlapping the live sample-logits tensor requires.

## Gates

* `test_host_buffer_staging` — new, focused.
* `test_sampler`, `test_runner`, `test_sampling_metadata`, `test_logits_processors` — the seam's neighbours.
* `scripts/agent-preflight.sh`.
* NemotronH A3 (GB10) — the working path this must not break.

## Stop conditions

Stop and report if the unified arm is not byte-identical to the wrap it
replaces: the whole safety argument for GB10 is that the unified branch is the
same expression.

## Owed

Nothing. #1313 is fixed in this change.

## Now

ACTIVE — repair landed on `fix/sampler-host-logits-device-tag`.
