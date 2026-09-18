ID: ISSUE-LOCAL-01M2F0PQWGSCXG0N4951NF9DPZ
Title: Enable the existing quantized WMMA prefill kernels on gfx1100
Row: KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA3
State: CLOSED
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: 2026-09-13

## Problem

Current main compiles and dispatches its generic rocWMMA Q4_K and Q6_K prefill kernels only on gfx1200/gfx1201. The installed rocWMMA 2.2.1 implements the same 16x16x16 signed-int8 operation on gfx1100. Verify admission-only reuse on physical gfx1100, preserve the attention architecture guard, and prove production reachability, numerical correctness, and same-binary prefill performance before accepting the default. The user selects RDNA3 WMMA as the next work and explicitly deprioritizes the unrelated #2773 characterization campaign.

## Resolution

13 September 2026: The landing change admits physical gfx1100 through the existing Q4_K and Q6_K WMMA dispatch. Independent review and operator verification pass the architecture, 240 original matrix, and public 1024-logit gates. Eight exact native prompt arrays and all generated IDs in six native processes match both task-pinned oracles. Native traces contain 152 WMMA calls per prefill and zero during decode. The observed prefill ratio is 1.2891 under dynamic clocks. The report and linked archive at docs/bench-evidence/rocm-rdna3-quant-wmma/README.md retain every value, ratio, failed attempt, and measurement limit. ISSUE-LOCAL-01M2F4WCD6ZK5VH5S8TF83APD6 remains open for whole-model floors and accepted clock attribution.
