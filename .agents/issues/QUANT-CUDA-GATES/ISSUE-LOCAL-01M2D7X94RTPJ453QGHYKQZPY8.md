ID: ISSUE-LOCAL-01M2D7X94RTPJ453QGHYKQZPY8
Title: The W8 keep-quant MoE arm is default-ON on Tenstorrent with no suite that runs there
Row: QUANT-CUDA-GATES
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

The W8 keep-quant MoE arm (`MoeBlockKqDevice`, `qwen3_5.cpp`) selects on
`vt::OpRegistered(..., d.q.device.type)` rather than on a device, which is what
`check-device-leakage.py` requires of that layer. Tenstorrent registers all
three ops it asks for -- `kMatmulBTQuantGrouped`, `kMoeSiluMul` and `kCastBf16`
at `tenstorrent_ops.cpp:8298, 8307, 8309` -- so the arm is DEFAULT-ON there. No
suite in this tree runs a MoE block on Tenstorrent and the fleet has no
Tenstorrent device to run one on, so the arm's byte-identity bar has never been
measured on that backend. `tests/vllm/models/test_qwen35_moe_kq_device.cpp`
gates CPU, CUDA and ROCm.

THE bf16 PRECEDENT ARM SHARES THE PREDICATE SHAPE BUT NOT THIS EXPOSURE, and
that was checked rather than assumed. `MoeBlockBf16Cuda` needs
`kMoeGroupedGemmBf16` and `kMoeGroupedGemmBf16GateUpSilu`, which Tenstorrent
does NOT register (`grep kMoeGroupedGemmBf16 src/vt/tenstorrent/` is empty) and
ROCm does (`rocm_ops.hip:292, 295`). So this is one arm's gap on one backend,
not a class.

AN EARLIER VERSION OF THIS PARAGRAPH ADDED "and on ROCm,
`tests/vllm/models/test_rocm_moe_bf16.cpp` gates it". THAT HALF IS WITHDRAWN, and it
was wrong in two independent ways. That suite never names `MoeBlock` or
`RunMoeBlock`, so it gates no bf16 ARM PREDICATE at all -- its
`kMoeGroupedGemmBf16` assertions (`test_rocm_moe_bf16.cpp:400-401`) are op-table
registration checks. And they sit inside
`TEST_CASE(... * doctest::skip(FixtureAbsent()))` (`:303-304`), which is inert
unless `VT_ROCM_MOE_FIXTURE` names an exported checkpoint and `VT_ROCM_MOE_ORACLE`
an oracle, so on an ordinary ROCm box they do not run either. The decisive half
of the paragraph -- the Tenstorrent `grep` -- is unaffected: it alone is what makes
this one arm's gap.

What is owed is a Tenstorrent case behind a skip plus a device to run it on, or
a narrowing of the predicate to the backends a suite gates -- and a narrowing
has to answer `check-device-leakage.py`, which is why W8 did not simply do it.

## Resolution

-
