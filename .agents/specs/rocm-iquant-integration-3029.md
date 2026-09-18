# Integrate the ROCm I-quant contribution and test loader admission

Row: `KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT`.
Issue: [#1940](https://github.com/mudler/vllm.cpp/issues/1940), which remains
open for the other quantized formats.
Contribution: [#3029](https://github.com/mudler/vllm.cpp/pull/3029).
Parent spec: [ROCm I-quant port](kernel-quant-ciq-gemm-rocm-iquant.md).
Contribution base: `7aa0aa00a8eb79d53e65685e78e8da6d3f10482a`.
Initial integration target: `415d17859500caf2a4cac00511820e4f4760e86f`.
Final integration target: `08a34c3a74d78046f83886f242d07110a70ff45e`, which
includes the prerequisite README scan repair from #3064.

## Scope and source

Resolve the three integration conflicts in `gguf_keep_quant.cpp`,
`rocm_grouped_gemm.hip`, and `test_backend_cross_device.cpp`.
Preserve both the target's behavior and the contribution's IQ4_XS and IQ3_XXS
admission, dense kernels, grouped kernels, and tests.
Do not import #3036 or redesign a kernel.

The parent spec defines the source algorithms and device gates.
`git log -S kIQ3_XXS -- src/vllm/model_executor/model_loader/gguf_keep_quant.cpp`
identifies `acd7d457d` as the loader admission change.
The contributor's routing test explicitly omits IQ3_XXS. Its direct operation
tests cannot detect deletion of IQ3_XXS from loader admission.

## Design and tests

Add a test through `RouteGgufTensor` for ROCm IQ3_XXS matrix and stacked
expert weights. Both aligned roles must keep their blocks. Ragged shapes,
disabled keep-quant, and CPU-reference mode must still expand to bf16.
Keep the broad existing routing table unchanged except for its obsolete
coverage comment. The new test owns the formerly missing admission guarantee.

Before accepting the regression, remove IQ3_XXS from the ROCm admission arm
in a scratch copy and require the test to fail. Restore the original source
and require the loader suite to pass.
Run focused loader and device-fit tests plus `scripts/agent-preflight.sh`.
The operator builds HIP and runs the contributor's device tests under a lease.
Generic I-quant checks on Strix do not establish gfx1200 performance.

## Records and stop conditions

If a keyed record conflicts, start from its complete target version and reapply
only this row's edit. Verify unrelated keys against the target byte-for-byte.
No lifecycle change or new benchmark publication belongs to this repair.
Stop if conflict resolution requires choosing between incompatible behaviors,
changes residency-budget semantics, or needs a new kernel design.

## Now

DONE, and partly superseded. The integration this spec defines landed against
target `08a34c3a7`. Main then moved the ROCm quant-GEMM registration into
`src/vt/rocm/rocm_quant_dot.hip` and shipped IQ3_XXS there itself, so the
second 2026-09-09 reconciliation dropped IQ3_XXS from this row. The
IQ3_XXS loader-admission test this spec asked for is kept and still passes,
now against main's kernel. See "Reconciliation with main" in the parent spec.
