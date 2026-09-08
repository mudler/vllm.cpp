# Seal every ROCm I-quant table against its CPU reference

Row: `KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT`.
Issue: [#3067](https://github.com/mudler/vllm.cpp/issues/3067).
Contribution: [#3029](https://github.com/mudler/vllm.cpp/pull/3029).
Parent spec: [ROCm I-quant port](kernel-quant-ciq-gemm-rocm-iquant.md).
Repair base: `b2ee9d8389caf974f3178613a6313e788dd93c4b`.

## Gap and source

The parent spec and `rocm_quant_iq_tables.h:17-20` require a complete seal.
Review found no executing comparison of the four ROCm device tables against
their CPU references. Host-parsed arrays currently agree, but that inspection
does not pin every executing device byte in a regression test.

Mirror the existing CUDA snapshot in `cuda_quant_dot.cu:2637` and the test in
`test_cuda_quant_dot.cpp:1702`. The CPU tables carry the parent spec's pinned
llama.cpp reference. This change adds no quantization algorithm or oracle.

## Design and scope

Add a HIP-free internal snapshot declaration for the four arrays:
`d_kmask_iq2xs`, `d_ksigns_iq2xs`, `d_iq3xxs_grid`, and `d_kvalues_iq4nl`.
Define the copy in `rocm_grouped_gemm.hip`, which defines the device symbols.
Use `hipMemcpyFromSymbol` and the existing HIP error checker. Compile-time
extent checks prevent truncation. Do not change table values, storage classes,
arithmetic, dispatch, or any other quantized format.

Add a HIP test to `test_backend_cross_device.cpp`. Compare each complete
snapshot array with its CPU reference using `memcmp`. Check all four extents
and the number of comparisons. Do not substitute host literals for device
copies. Follow the executable's missing-backend convention; a skipped device
case is not device evidence.

## Tests and gates

Commit the test and interface before the copy implementation. The missing
implementation is the initial compile/link gap. The operator then executes
the completed seal under a HIP lease and mutates one entry in each of the four
device tables separately. Every mutation must fail its named comparison;
restore the table byte-for-byte between runs. Deleting a snapshot copy must
also fail, since a seal must observe each symbol rather than compare nothing.

Run the focused CPU loader and device-fit tests and the complete host
preflight. Host builds do not establish HIP correctness. Run the existing
I-quant numerical device gates with the seal under the operator's lease.
Independent scoped review and the operator's own gate remain required.

## Risks and stop conditions

A passing tolerance-based dot test can miss an unvisited table entry. The
byte-exact seal closes only that gap, not gfx1200 performance or the remaining
formats owned by #1940. Keep that broader issue open.
Stop if the registered HIP implementation cannot expose its actual device
symbols, or if source/oracle disagreement requires changing table values.

## Now

ACTIVE: the device-byte seal is specified before its test and implementation.
