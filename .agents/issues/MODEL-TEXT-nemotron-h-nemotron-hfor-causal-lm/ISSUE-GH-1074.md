ID: ISSUE-GH-1074
Title: The NemotronH model-matrix row described `main` as of 2026-08-12 and stayed `INVENTORIED` while A2-R, A2-P, A2-Q2a and the A3 driver landed on top of it. Reconciled against `main` at `b626be75a`, every claim re-verified rather than inherited: the non-gated `relu²` MoE EXISTS (`4d0c399e1`, `vt::MoeRelu2` called from `nemotron_h.cpp:354`), ModelOpt `MIXED_PRECISION` per-module loading EXISTS (`1bc5ef82c`), the MTP head is STILL OWED (W5, 270 tensors deferred by name), and the `KERNEL-SSM-MAMBA` block is FALSE — [#496](https://github.com/mudler/vllm.cpp/issues/496) W1 landed the host arm at `47960a009` and W2 the CUDA arm at `43a6c5518`, in `src/vt/cuda/cuda_mamba2_ssd.cuh`, a `.cuh` included by `cuda_gdn.cu` rather than a translation unit of its own, which is why a `src/vt/*mamba*` FILE GLOB reports absence; `nemotron_h.cpp:597` calls that op today. Row moved `INVENTORIED` -> `PARTIAL` with the rollup, the checklist entry and the projections it owes. `PARTIAL` and not `ACTIVE` deliberately: `check-agent-record.py` requires an `ACTIVE` row to name a `CLAIM-*` row a claim source carries, no file under `.agents/claims/` claims this row, and authoring one for another session's in-flight work would be a fabricated record. NO end-to-end token gate has passed and no throughput, latency or memory number is claimed; the A3 gate stays PENDING and [#1157](https://github.com/mudler/vllm.cpp/issues/1157) is the open decode divergence
Row: MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 1074
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:480`

### Frozen archive evidence

> | [#1074](https://github.com/mudler/vllm.cpp/issues/1074) | `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` | The NemotronH model-matrix row described `main` as of 2026-08-12 and stayed `INVENTORIED` while A2-R, A2-P, A2-Q2a and the A3 driver landed on top of it. Reconciled against `main` at `b626be75a`, every claim re-verified rather than inherited: the non-gated `relu²` MoE EXISTS (`4d0c399e1`, `vt::MoeRelu2` called from `nemotron_h.cpp:354`), ModelOpt `MIXED_PRECISION` per-module loading EXISTS (`1bc5ef82c`), the MTP head is STILL OWED (W5, 270 tensors deferred by name), and the `KERNEL-SSM-MAMBA` block is FALSE — [#496](https://github.com/mudler/vllm.cpp/issues/496) W1 landed the host arm at `47960a009` and W2 the CUDA arm at `43a6c5518`, in `src/vt/cuda/cuda_mamba2_ssd.cuh`, a `.cuh` included by `cuda_gdn.cu` rather than a translation unit of its own, which is why a `src/vt/*mamba*` FILE GLOB reports absence; `nemotron_h.cpp:597` calls that op today. Row moved `INVENTORIED` -> `PARTIAL` with the rollup, the checklist entry and the projections it owes. `PARTIAL` and not `ACTIVE` deliberately: `check-agent-record.py` requires an `ACTIVE` row to name a `CLAIM-*` row a claim source carries, no file under `.agents/claims/` claims this row, and authoring one for another session's in-flight work would be a fabricated record. NO end-to-end token gate has passed and no throughput, latency or memory number is claimed; the A3 gate stays PENDING and [#1157](https://github.com/mudler/vllm.cpp/issues/1157) is the open decode divergence | bug |

## Resolution

-
