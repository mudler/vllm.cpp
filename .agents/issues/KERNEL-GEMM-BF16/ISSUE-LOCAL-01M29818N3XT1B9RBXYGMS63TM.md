ID: ISSUE-LOCAL-01M29818N3XT1B9RBXYGMS63TM
Title: The draw survey cannot reach the CUTLASS W4A4 tactic path on any loadable staged checkpoint, so it collects zero draws
Row: KERNEL-GEMM-BF16
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-11
Updated: 2026-09-11
Closed: -

## Problem

The survey exists to measure NVFP4 W4A4 CUTLASS tactic selection (#2750, #2751, #2752). On every checkpoint currently staged it measures nothing, for two compounding reasons found by running it on a device.

FIRST, the checkpoint it was pointed at cannot load. `qwen38-27b-nvfp4` (unsloth) is compressed-tensors `mixed-precision`: its `group_0` declares a PER-OUTPUT-CHANNEL weight scale with DYNAMIC per-token activations, which `LoadFp8Raw` cannot represent, so 233 modules refuse by name at `qwen3_5_dense_weights.cpp`. This is already documented in `docs/USAGE.md` ("NVFP4 modules load | FP8 modules and quantized KV cache are refused") and was an operator error in choosing it. The smoke run died at phase [P] with `draw00: rc=1 algo_keys=0 plans=0 cache=None`.

SECOND, and the actual blocker: the loadable alternative does not reach the tactic path at default settings. `qwen38-27b-nvfp4-radixark` loads and decodes on the device (12.22 tok/s per-stream, 20992 tokens, repeat agrees at 12.21) and its FP8 half is per-tensor STATIC, so nothing refuses. But its 193 NVFP4 modules route to the W4A16 arm, because `Nvfp4Weight::alpha` stays 0 unless `VT_MODELOPT_W4A4=1` consumes the shipped `input_scale`. `dense_nvfp4_gemm.h` states its own scope: "W4A16 ONLY. This header deliberately does NOT carry the true-W4A4 (fp4-ACTIVATION) path". The W4A16 arm runs Marlin; `SelectPlan` and the plan cache live in `cuda_matmul_nvfp4_cutlass.cu` on the W4A4 side. So a default run tunes nothing and the harness refuses with its own exit 73/70 preconditions.

Verified from the tensor index rather than prose: radixark ships 401 `input_scale`, 401 `weight_scale`, 193 `weight_scale_2` across 2194 tensors, and `layers.0.mlp.gate_proj` carries both `weight_scale_2` and `input_scale`. So the opt-in can flip `IsTrueW4A4()`, which is what `uses_nvfp4_w4a4()` reads.

WHAT IS OWED HERE. `run_draw` sets exactly six environment variables and none is `VT_MODELOPT_W4A4`; neither the driver nor the judge mentions it. The arm must be SET and never inherited, exactly as the tactic-set arm already is, stamped into PROVENANCE and into every draw record, and refused when mixed within one evidence root. A draw whose activation arm was decided by the operator's ambient environment is a draw nobody can attribute.

ORDERING. Enabling the arm is not this row's decision to make alone: `#2760` on row QUANT-QWEN38-27B-NVFP4-ARM owns whether `VT_MODELOPT_W4A4=1` is CORRECT, and its ask 2 designs the five-arm experiment that answers it. Tactic draws taken on an activation path that has not been shown token-exact would measure a defect rather than a tactic. This issue delivers the capability; the decision to trust its output waits on that experiment.

## Resolution

-
