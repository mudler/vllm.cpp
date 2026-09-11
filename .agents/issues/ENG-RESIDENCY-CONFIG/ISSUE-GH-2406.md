ID: ISSUE-GH-2406
Title: quant_repack is set from a host-ISA probe with no device term: aarch64 i8mm + --device cuda stages ARM-repacked weights the CUDA kernel cannot read
Row: ENG-RESIDENCY-CONFIG
State: OPEN
Kind: UNKNOWN
GitHub: 2406
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `-`
>
> Owed by `.agents/specs/gguf-residency-resolved-device.md` under `## Owed`.
>
> `GgufLoadPolicy::FromEnv` (`src/vllm/model_executor/model_loader/gguf_keep_quant.cpp`) sets:
>
> ```cpp
> p.quant_repack = p.keep_quant && !p.cpu_ref && vt::cpu::QuantRepackActive();
> ```
>
> `vt::cpu::QuantRepackActive()` is a pure **host-ISA** query. It has no device term. Its sibling two lines below **does**:
>
> ```cpp
> p.elem_kn_repack = !p.cpu_ref && EnvOnOr("VT_CPU_ELEM_KN_REPACK", false) &&
>                    dev == vt::DeviceType::kCPU;
> ```
>
> and that gate exists for precisely this reason — its own comment says "a staged device would upload the transposed bytes and read them as [N,K]".
>
> **Consequence.** On an aarch64 i8mm host that also has a GPU — dgx GB10 and Jetson Thor, both fleet devices — a `--device cuda` load of a Q8_0 GGUF sets `quant_repack = true`. The weights are repacked at load into `block_q8_0x4` (an ARM `mmla` tile layout) and then staged to the card. `grep -c repacked src/vt/cuda/cuda_quant_dot.cu` returns **0**: there is no reader for the marker on the CUDA side.
>
> This is the same shape as the W5r defect (#2031), where a shared `ResidentWeight` dropped the load-time repack markers.
>
> **Not certain to be wrong in every case,** which is why this is an issue and not a patch. `cuda_quant_dot.cu:1841-1846` documents that CUDA falls back to the CPU kernel for encodings it lacks, and that fallback *would* read the repacked bytes correctly. So the gate may be a performance trade rather than a pure correctness fix, and it needs measuring on an aarch64 CUDA box before anyone changes the line. Both arms are untestable on an x86 CPU-only host.
>
> **Found while** landing #2392. A comment added there originally claimed "EVERY device-dependent decision in this struct reads this field", which is false because of this line; the fresh review of #2397 caught it. The comment now names this gap explicitly rather than asserting it away.
>
> **Repro (static):**
> ```sh
> grep -n 'quant_repack = ' src/vllm/model_executor/model_loader/gguf_keep_quant.cpp   # no dev term
> grep -c 'repacked' src/vt/cuda/cuda_quant_dot.cu                                     # 0
> ```

## Resolution

-
