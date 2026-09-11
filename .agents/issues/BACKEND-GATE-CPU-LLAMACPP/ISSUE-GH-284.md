ID: ISSUE-GH-284
Title: perf(cpu): close Raspberry Pi 5 A76 BF16 GEMM gap against llama.cpp
Row: BACKEND-GATE-CPU-LLAMACPP
State: OPEN
Kind: UNKNOWN
GitHub: 284
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-10
Updated: 2026-08-10
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Row
>
> `KERNEL-GEMM-CPU-ELEM-A76`
>
> ## Baseline
>
> The landed Raspberry Pi 5 / Cortex-A76 lane (#65, #79) is correct but does not meet the same-file llama.cpp floor for Qwen3.5-2B Q8_K_XL on four A76 cores:
>
> - prefill: 12.81 vs 27.77 tok/s (`0.461x`)
> - decode: 2.55 vs 3.91 tok/s (`0.653x`)
> - output-equivalent E2E: 2.46 vs 3.77 tok/s (`0.653x`)
> - peak RSS: 2.841 vs 3.747 GiB (`0.758x`, already better)
>
> The last clean profile attributes 57.76% of the model run to the BF16 `Bt16Neon` elementwise GEMM. This is the largest measured gap and the next lever named by the landed RPi5 campaign spec.
>
> ## Scope
>
> 1. Reproduce the current main baseline and profile both vllm.cpp and llama.cpp on the identical workload.
> 2. Add a focused microbenchmark and PMU/disassembly evidence for the reached BF16/F16/F32 elementwise GEMM shapes.
> 3. Optimize the C++/NEON implementation first, recursively gating the kernel, enclosing model phase, and full end-to-end workload.
> 4. Do not write or select assembly until the C++ path is beyond llama.cpp on prefill, decode, and E2E without regressing correctness or the existing RSS win.
> 5. Once that C++ floor is closed, use compiler disassembly plus Pi PMU evidence to decide whether an AAPCS64 implementation has measurable remaining headroom.
>
> ## Constraints
>
> - Build and test AArch64 artifacts locally under QEMU; do not compile on the Pi.
> - Use `rich@rpi5fan.lan` only for execution and physical-PMU measurements.
> - Same GGUF bytes, prompt/token counts, four-core affinity, sampling, and llama.cpp pin as the existing binding evidence.
> - Preserve token/output correctness and measure every recursive enclosing scope after each accepted optimization.
> - No dependency or package installation and no large asset download.
>
> ## Done when
>
> - C++/NEON vllm.cpp is faster than llama.cpp on the binding prefill, decode, and E2E axes, with RSS no worse than the landed baseline and repeated idle same-binary evidence.
> - Any later assembly has mutation-tested correctness, ABI/disassembly proof, PMU evidence, and a recursive model win over that already-parity C++ baseline.
>

## Resolution

-
