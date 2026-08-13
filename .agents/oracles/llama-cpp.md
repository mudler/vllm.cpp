# llama.cpp — the CPU and GGUF k-quant floor

Never the mirror source: llama.cpp's structure is not vLLM's, and a behavior
difference between them is settled by vLLM. What it supplies is a **floor** —
the CPU and GGUF k-quant speed and memory numbers a user can actually get today,
which is the honest denominator on every path where vLLM's own CPU support is
not the thing being compared.

The pin is a **local fork**, not upstream `master`, because the CPU floor
campaign builds it with a fixed recipe and reads its kernels; the fork is what
`237ad9b96` names. Measured evidence and the build recipe are in
[`../specs/cpu-llamacpp-floor-remeasure-2026-07-22.md`](../specs/cpu-llamacpp-floor-remeasure-2026-07-22.md),
which carries the raw per-run numbers, and the same pin backs the A76 dot-product,
elementwise-GEMM, GDN-orientation and threadpool specs.

```oracle-pin
id = llama-cpp
role = secondary
upstream = https://github.com/ggml-org/llama.cpp
scope = CPU and GGUF k-quant speed and memory floors, quant-matched against the same weights
pin = 237ad9b96
pin_label = b9892
pinned_on = 2026-07-22
gateable = yes
evidence = .agents/specs/cpu-llamacpp-floor-remeasure-2026-07-22.md
```
