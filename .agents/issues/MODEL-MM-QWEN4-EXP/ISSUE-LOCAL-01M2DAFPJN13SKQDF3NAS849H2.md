ID: ISSUE-LOCAL-01M2DAFPJN13SKQDF3NAS849H2
Title: qwen4_exp keeps the 27.5 GiB PLE table device-resident where llama.cpp keeps it host-resident
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: gap
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

llama.cpp at the llama-cpp-qwen4exp pin routes exactly one tensor, per_layer_token_embd.weight (iq4_nl, 27,465.95 MiB), to a host CPU buffer and reads it only with a GET_ROWS gather. vllm.cpp's ROCm arm holds the whole 67.55 GiB artifact on device. This issue owns establishing the cost of that difference on gfx1151 and scoping a per-tensor residency policy. It does NOT own a cross-engine ratio: the qwen4_exp ROCm arm has no declared token-exact gate (ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG).

## Findings, 2026-09-13

MEASURED on strix:gpu0, rc job 97b5ccb8, evidence
docs/bench-evidence/qwen4exp-ple-placement-gfx1151-20260913.md.

llama.cpp is host-resident for this tensor because its HIP backend REFUSES
GET_ROWS on an IQ4_NL source whose row length is not a multiple of QK_K
(ggml/src/ggml-cuda/ggml-cuda.cu:5012-5016 at pin 035e2273; this row is 160,
QK_K is 256). Forcing it to ROCm0 makes ggml_backend_sched run the gather on the
CPU backend and re-copy the whole 28.80 GB tensor into a host compute buffer on
every graph evaluation: the reserved buffer reads 27,467.24 MiB and decode falls
from 25.4671 to 1.9988 tok/s, an implied 62.5 GB/s device-to-host copy.

So there is NO hybrid placement to port. vllm.cpp already implements the kernel
llama.cpp lacks (src/vt/rocm/rocm_embedding_quant.hip:51,142, registered at
src/vt/rocm/rocm_ops.hip:207, #3093), which carries no QK_K assumption.

The proposal that survives is a per-tensor host-residency POLICY, scoped in
.agents/specs/qwen4-exp-ple-host-residency.md, whose predicted throughput gain
is ZERO and whose case is 26.822 GiB of device memory plus letting a device
without a block gather load instead of refuse.

## Still owed by this issue

- A ranked kernel table for the vllm.cpp ROCm arm on gfx1151. We own no per-kernel
  timing instrument on ROCm (src/vt/rocm/rocm_backend.hip:460), and rocprofv3
  produced invalid timestamps on this board (#3040). Where our decode step time
  goes on this model is UNVERIFIED.
- A timed H1 control. The expert towers are 38,341.02 MiB against 30 GiB of host
  RAM on this worker, so the experts-to-host probe OOMed before it timed anything.
- The combined `per_layer_token_embd=CPU;.*=ROCm0` leg. The follow-up lease
  refused because the 67.55 GiB artifact is no longer staged on the worker.

## Resolution

-
