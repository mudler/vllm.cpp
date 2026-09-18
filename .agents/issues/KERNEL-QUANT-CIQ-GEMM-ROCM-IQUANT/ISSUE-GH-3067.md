ID: ISSUE-GH-3067
Title: test(KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT): seal every device table entry
Row: KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT
State: CLOSED
Kind: UNKNOWN
GitHub: 3067
Mirror: SYNCED
Availability: FULL
Created: 2026-09-08
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT`
>
> PR #3029 copies I-quant lookup tables into ROCm constant memory. Its owning spec, `.agents/specs/kernel-quant-ciq-gemm-rocm-iquant.md`, requires a complete table seal. The new header also explicitly requires a device snapshot and byte comparison. The changed tests do not implement that gate.
>
> Fresh review of integration head `b2ee9d8389caf974f3178613a6313e788dd93c4b` found that all four host-parsed arrays currently match their CPU references. This does not verify the executing device bytes or pin every entry in a committed test.
>
> Add the ROCm equivalent of the existing CUDA device-table snapshot and compare every entry against the pinned oracle reference. Preserve the current table values, kernel arithmetic, and dispatch. Mutating one entry in each table must make the focused device seal fail. Run independent review and the operator's leased ROCm gate before merging #3029. Keep the other formats owed by #1940 open.
>

## Resolution

Fixed by PR #3029: IQ4_XS and IQ3_XXS now have ROCm DotIQ4XS/DotIQ3XXS kernels in rocm_grouped_gemm.hip, admitted in DeviceKeepQuantSupported.
