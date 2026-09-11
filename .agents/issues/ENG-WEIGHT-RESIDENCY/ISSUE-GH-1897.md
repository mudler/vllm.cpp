ID: ISSUE-GH-1897
Title: Weights are never made device-resident at load
Row: ENG-WEIGHT-RESIDENCY
State: OPEN
Kind: UNKNOWN
GitHub: 1897
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-25
Updated: 2026-08-25
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What's broken
>
> `vllm_engine_load` returns with **no weights resident on the device**. The first forward then uploads the model through `ResidentWeight`'s lazy `if (!w.d_dev)` branch: one `cudaMemcpyAsync(cudaMemcpyDefault)` per weight per layer, from **pageable** host sources (the read-only mmap of the GGUF file). On a CMP 170HX (GA100, sm_80) this is **37.3 GB across 405 copy calls inside the first request's forward**.
>
> Measured in one process, two passes over the same prompt (2048-token chunk, `Qwen3.8-27B-UD-Q8_K_XL.gguf`):
>
> | Pass | Wall | H2D copied |
> |---|---:|---:|
> | 1 (cold weights) | 137.2 s | 37.3 GB |
> | 2 (weights resident) | 117.0 s | 0.1 GB |
>
> The 20.2 s difference is the upload. At this host's PCIe gen2 x4 link (1.7 GB/s measured) the copy storm is the dominant term of first-request latency; at a full x16 link the same 37.3 GB still costs ~5 s of first-request latency that belongs at load time. Upstream vLLM pays this cost during weight loading, never during a request.
>
> ## Reproduce
>
> Any discrete-GPU load; the effect is easiest to see on a slow link.
>
> ```sh
> build-cuda/examples/vllm-cli --model <27B gguf> --prompt "x" --max-tokens 4
> # first request: ~25 s wall at batch 1 before the first token
> ```
>
> Or hook the copy calls (LD_PRELOAD interposer on `cudaMemcpyAsync`, `kind==cudaMemcpyHostToDevice||cudaMemcpyDefault`): the copies appear only on the first forward, ~405 calls, sources resolving in `/proc/<pid>/maps` to the GGUF's read-only file mapping.
>
> ## Root cause: two defects, one symptom
>
> 1. **Residency is lazy, not load-time.** `vllm_engine_load` builds the engine, resolves the KV pool, and returns. Every `ResidentWeight` call in `src/vllm/model_executor/models/qwen3_5.cpp` (the `if (!w.d_dev)` branch, ~line 1280) performs alloc+upload on first touch inside `execute_model`. Nothing warms it at load.
>
> 2. **Staging is pageable.** `CudaBackend::Copy` (`src/vt/cuda/cuda_backend.cu:89-93`) explicitly tolerates pageable pointers by passing them to `cudaMemcpyAsync`. A pageable source is not a DMA source: the driver stages it through a pinned bounce buffer, and the submission call itself blocks while the staging copy runs. Measured standalone on this host: a 94 MB H2D from a pageable (file-mmap) source blocks the submitting call ~57 ms (about 1.7 GB/s); the same size from a pinned buffer submits in ~2 µs with the DMA running at the same link rate. Upstream vLLM (through torch) stages from pinned memory.
>
> The pageable tolerance is documented in-tree as intentional for unified-memory hosts, where handing a host pointer to a device kernel is legal. On a discrete GPU the tolerance turns every lazy upload into a synchronous, staged copy.
>
> `VT_QWEN35_ALIAS_HOST_WEIGHTS` has no effect on a discrete device (the predicate answers false by design), so there is no current escape hatch.
>
> ## Fix direction
>
> 1. Warm weight residency inside `vllm_engine_load`: after the model object is built, walk its weights and touch each `ResidentWeight` once, so the upload cost lands at load where it belongs. The upload accounting already exists (`vllm::load_stats::AddDeviceUpload`), so the load-time path can be gated and measured with it.
> 2. Stage through a pinned ring buffer in `CudaBackend::Copy` when the source is pageable and the device is discrete: a fixed-size pinned buffer + `cudaMemcpyAsync` in chunks keeps submissions asynchronous and lets the ring overlap with allocation of the next weight.
>
> Either half alone helps. Together, they make first-request latency a function of the link rate rather than of synchronous staging walks.
>
> ## Scope and non-goals
>
> - This issue is the upload path only. A separate residual exists on this specific host's driver (610.57.04): a subset of kernel launches block in a userspace spin inside `libcuda.so`'s green-context/topology enumeration, worth ~70 s of the 117 s resident-weights pass. That does not reproduce standalone and is reported separately once isolated.
> - Decode is unaffected (no H2D in steady decode; 24 tok/s warm on the same host).
> - `docs/USAGE.md` documents none of this, so no public doc is invalidated by the fix.
>
> ## Related issues (same family, different defects)
>
> - [#974](https://github.com/mudler/vllm.cpp/issues/974): the FP8 W8A8 helper skips `AddDeviceUpload`/`AdoptDeviceBytesAsHost`. The same `ResidentWeight` family, a bookkeeping gap rather than upload timing.
> - [#350](https://github.com/mudler/vllm.cpp/issues/350): staging disabled twice on GB10 by a wrong predicate. The unified-memory inverse of this report's discrete-GPU staging gap.
> - [#1312](https://github.com/mudler/vllm.cpp/issues/1312): NemotronH MoE re-uploads the router gate 46x per decode token. The recurring-H2D cousin, same class of unwanted host traffic.
> - [#1299](https://github.com/mudler/vllm.cpp/issues/1299): non-expert weights resident TWICE on a GB10 (first-forward OOM). Duplication rather than lazy upload.
>
> None of these covers "weights are never made resident at load, and the lazy path stages from pageable memory on discrete GPUs."
>
> ## Environment
>
> - GPU: NVIDIA CMP 170HX 64GB (GA100, sm_80, 70 SMs), driver 610.57.04, CUDA 13.3 (V13.3.73). PCIe gen2 x4 negotiated, this card's physical maximum: other lanes lack AC coupling capacitors
> - Link measurement: 1.7 GB/s H2D sustained; 94 MB pageable H2D blocks its submission ~57 ms, pinned submits ~2 µs (standalone probe, same host)
> - Model: `Qwen3.8-27B-UD-Q8_K_XL.gguf` (hybrid: 48 GDN + 16 full-attention layers), ~35.6 GiB resident after load
> - Tree: main @ 1724be38e, build `-DVLLM_CPP_CUDA_ARCHITECTURES=80`

## Resolution

-
