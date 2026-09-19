# Leaf spec: Vulkan TQ1_0 keep-quant matmul — ternary experts on-device

**Row:** `BACKEND-VULKAN-TQ1_0` (backend-matrix, leaf of `BACKEND-VULKAN`).
**Issue:** [#331](https://github.com/mudler/vllm.cpp/issues/331).
**Status:** `READY` — implementation complete, 42/42 tests pass.
**Upstream / port source:** llama.cpp `ggml/src/ggml-vulkan/` TQ2_0 pattern
(PR #25850) adapted for TQ1_0 ternary encoding; CPU reference in
`src/vt/cpu/cpu_quant_dot.cpp` `VecDotTQ1_0Q8_K`.

## Scope

- **In scope:** Vulkan compute shaders for TQ1_0 ternary weight matmul against
  Q8_K activations, both host-quantized and on-device-quantized paths; grouped
  (per-token expert) and non-grouped variants; fused gate+up+SwiGLU MoE kernel;
  committed SPIR-V for all new shaders; test coverage in
  `tests/vt/test_vulkan_backend.cpp`.
- **Out of scope:** TQ1_0 MMQ/int-dot path (same as TQ2_0 — not wired); Metal or
  CUDA TQ1_0 kernels; non-Vulkan backends.

## Background

TQ1_0 packs 5 base-3 digits per byte in `qs[48]` plus 4 per byte in `qh[4]`,
with a single f16 scale per 256-element block (54 bytes total). Trit extraction:
`q = byte * pow3[l]` (uint8 wrap); `xi = (q * 3) >> 8`; `w = (xi - 1) * d`,
giving `{-1, 0, +1}` scaled by `d`.

The maple 20B MoE model uses TQ1_0 expert weights. Without Vulkan TQ1_0
support, the model falls back to the CPU reference tier for every expert GEMM,
which is the bottleneck identified in the maple-vulkan-throughput task.

## Upstream chain

| Component | Upstream anchor | What it establishes |
|---|---|---|
| TQ2_0 Vulkan pattern | llama.cpp PR #25850 | shader structure, binding layout, spec constants |
| TQ1_0 block layout | `ggml/src/ggml-common.h` `block_tq1_0` | 54-byte block, qs[48]+qh[4]+f16 d |
| TQ1_0 trit extraction | `ggml/src/ggml-quants.c` `dequantize_row_tq1_0` | pow3 table, uint8 wrap, (q*3)>>8 |
| CPU vec_dot reference | `src/vt/cpu/cpu_quant_dot.cpp` `VecDotTQ1_0Q8_K` | arithmetic contract for the shader |

## Design

Five new shaders, mirroring the TQ2_0 set:

1. `vt_matmul_bt_tq1_0.comp` — host-quantized GEMV, Q8_K activations on host.
2. `vt_matmul_bt_tq1_0_grouped.comp` — grouped variant for per-token expert
   selection.
3. `vt_matmul_bt_tq1_0_dev.comp` — on-device Q8_K quantization + matmul (decode
   and prefill).
4. `vt_matmul_bt_tq1_0_grouped_dev.comp` — grouped on-device variant.
5. `vt_moe_gate_up_swiglu_grouped_tq1_0.comp` — fused gate+up+SwiGLU for the MoE
   expert path.

The host glue in `vulkan_ops.cpp` unifies TQ1_0 and TQ2_0 dispatch through the
same `TryNativeTQ2Decode` / `MatmulBTQuantKernelVulkan` /
`TryNativeTQ2Grouped` / `MatmulBTQuantGroupedKernelVulkan` /
`TryNativeMoeGateUpSwiGLUGroupedTQ2` / `MoeGateUpSwiGLUGroupedKernelVulkan`
paths, selecting the shader by weight dtype.

## Risks/decisions

1. **No MMQ path.** Same as TQ2_0 — the int-dot cooperative-matrix path is not
   wired for ternary types. The keep-quant GEMV and dequant-then-matmul paths
   cover decode and prefill.
2. **Committed SPIR-V.** The build machine has `glslc` (shaderc 2023.8) but the
   repo's hermetic build commits SPIR-V ahead of time. The generator script
   (`scripts/gen-vulkan-spirv.py`) compiles all shaders; the TQ1_0 SPIR-V was
   compiled and appended to `vulkan_spirv.cpp`/`vulkan_spirv.h`.
3. **NMSE tier.** The 128-lane K-split reduction uses a different accumulation
   order than the scalar CPU loop, so the result lands in the NMSE tolerance
   tier (same accepted trade as TQ2_0 and `vt_matmul_vec`).

## Tests

- `tests/vt/test_vulkan_backend.cpp`: TQ1_0 keep-quant matmul (M>=1
  decode+prefill), grouped matmul, and fused MoE gate+up+SwiGLU tests, all
  matching the CPU oracle.
- Module count assertion updated from 35 to 40 (5 new TQ1_0 + 3 TQ2_0 dev + 2
  MoE/rope that were already on the build tree).
- Spec-id assertions for all 5 new TQ1_0 shaders.

## Gates

- `test_vulkan_backend`: 42/42 pass, 2299/2299 assertions.
- TQ1_0 maple model: "The capital of France is" → "Paris. Paris is known for
  the Eiffel Tower..." at 5.1 tok/s on Intel Arc Pro B60.
- TQ2_0 maple model: same output, same throughput (no regression).

## Evidence

- `test_vulkan_backend`: 42 test cases, 2299 assertions, 0 failures.
- TQ1_0 model: 5.057 tok/s, correct "Paris" output.
- TQ2_0 model: 4.874 tok/s, correct "Paris" output (no regression).
- llama.cpp `test-backend-ops`: MUL_MAT, MUL_MAT_ID, DEQUANT, GET_ROWS all pass
  for `type_a=tq1_0` on the same hardware.

## Outcome

TQ1_0 Vulkan support landed. The maple 20B MoE model runs on Vulkan at 5.1 tok/s
(11x improvement from the 0.46 tok/s CPU-fallback baseline). Both TQ1_0 and
TQ2_0 models produce correct output. The fused MoE gate+up+SwiGLU kernel
eliminates the per-expert dispatch overhead that caused the original bottleneck.
