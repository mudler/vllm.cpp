ID: ISSUE-GH-1189
Title: Block-wise (128x128) FP8 so `Qwen/Qwen3.8-27B-FP8` runs instead of being refused. `weight_block_size` appears nowhere in `src/` or `include/`, so `469f38395` refuses the arm by name (#1166). Six independently landable milestones; M1 lands here. M1 is `vt::QuantFp8Group`, the dynamic per-token per-group activation quant, CPU and CUDA. The numerics mirror the kernel that ACTUALLY EXECUTES on a CUDA-alike platform with a contiguous input, which is the C++ custom op at `csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu:42-96` and NOT the Triton kernel at `fp8_utils.py:95-150`: `fp8_utils.py:635-650` calls the former and returns before the latter. The two arms differ, and the difference is measurable rather than cosmetic. The CUDA kernel divides twice, `local_absmax / max_8bit` at `:68` and `static_cast<float>(src) / y_s` at `:85`; the Triton kernel multiplies by `(1.0 / fp8_max)` at `fp8_utils.py:145` under a comment that names the 1-ULP gap. Upstream's own test tolerates the gap with `rtol=0.15` (`test_block_fp8.py:112-114`), so a value comparison cannot tell the two apart and only a byte comparison against a spelled-out reference can. `eps` is the reduction's INITIAL value (`:47`), not a post-clamp, which is what keeps an all-zero group from dividing by zero. Scope refused here and owed to later milestones: the block-scaled GEMM (M2), `Fp8BlockWeight` and the loader (M3), `Fp8BlockLinearMethod` and the Qwen3.5 wiring (M4), the mainloop-scaled CUTLASS kernel and the column-major/TMA-aligned scale layouts (M5), merged `gate_up`/QKV (M6). M1 lands UNREACHED: no production entry point dispatches `vt::QuantFp8Group` at its merge commit, M4 owns the wiring, and `.agents/specs/vt-quant-fp8-group.md` lists it under `## Owed`. The CUDA arm compiles and its on-hardware leg is owed too, because the row took no GPU lease by design: the CPU arm is the gateable one
Row: VT-QUANT-FP8-GROUP
State: UNKNOWN
Kind: feature
GitHub: 1189
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:383`

### Frozen archive evidence

> | [#1189](https://github.com/mudler/vllm.cpp/issues/1189) | `VT-QUANT-FP8-GROUP` | Block-wise (128x128) FP8 so `Qwen/Qwen3.8-27B-FP8` runs instead of being refused. `weight_block_size` appears nowhere in `src/` or `include/`, so `469f38395` refuses the arm by name (#1166). Six independently landable milestones; M1 lands here. M1 is `vt::QuantFp8Group`, the dynamic per-token per-group activation quant, CPU and CUDA. The numerics mirror the kernel that ACTUALLY EXECUTES on a CUDA-alike platform with a contiguous input, which is the C++ custom op at `csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu:42-96` and NOT the Triton kernel at `fp8_utils.py:95-150`: `fp8_utils.py:635-650` calls the former and returns before the latter. The two arms differ, and the difference is measurable rather than cosmetic. The CUDA kernel divides twice, `local_absmax / max_8bit` at `:68` and `static_cast<float>(src) / y_s` at `:85`; the Triton kernel multiplies by `(1.0 / fp8_max)` at `fp8_utils.py:145` under a comment that names the 1-ULP gap. Upstream's own test tolerates the gap with `rtol=0.15` (`test_block_fp8.py:112-114`), so a value comparison cannot tell the two apart and only a byte comparison against a spelled-out reference can. `eps` is the reduction's INITIAL value (`:47`), not a post-clamp, which is what keeps an all-zero group from dividing by zero. Scope refused here and owed to later milestones: the block-scaled GEMM (M2), `Fp8BlockWeight` and the loader (M3), `Fp8BlockLinearMethod` and the Qwen3.5 wiring (M4), the mainloop-scaled CUTLASS kernel and the column-major/TMA-aligned scale layouts (M5), merged `gate_up`/QKV (M6). M1 lands UNREACHED: no production entry point dispatches `vt::QuantFp8Group` at its merge commit, M4 owns the wiring, and `.agents/specs/vt-quant-fp8-group.md` lists it under `## Owed`. The CUDA arm compiles and its on-hardware leg is owed too, because the row took no GPU lease by design: the CPU arm is the gateable one | feature |

## Resolution

-
