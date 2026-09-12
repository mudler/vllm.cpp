ID: ISSUE-LOCAL-01M29KF3SNQCR8163840XSBK5S
Title: test_deepseek_v4_mm_chat's image branch encodes a CPU-only premise and can never hold on a CUDA build
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

The image branch of test_deepseek_v4_mm_chat asserts that the served error names 'W7-device', which is kDevicePending. DeepseekV4Model::ForwardDevice emits that message only from VT_CHECK(deepseek_v4::V4DeviceKernelsAvailable(), kDevicePending), a predicate that is FALSE exactly when the V4 device kernels are ABSENT. On any CUDA build carrying those kernels the refusal therefore cannot fire and the assertion can never hold, so the case encodes the CPU-only build as a premise rather than testing a behaviour. Measured on thor:gpu0 during W7-CUDA: one assertion of 650 fails for this reason, while the sibling deepseek_v4.cpp check passes because the new message names that file. Owed: a device-aware expectation that states what a CUDA build must answer, rather than a CPU-shaped one. See .agents/specs/deepseek-v4-flash-vision.md section 'W7-CUDA evidence'.

## Resolution

2026-09-12 REPAIRED on this row's branch, and the CUDA expectation is MEASURED rather than assumed. The image branch now selects on vllm::deepseek_v4::V4DeviceKernelsAvailable() - the same symbol test_cuda_deepseek_v4.cpp uses, so the two suites agree on what 'this build carries the V4 device kernels' means. Without the kernels the case keeps the old expectation, because kDevicePending is exactly the refusal ForwardDevice's first guard emits there. With them, the kDevicePending refusal cannot fire at all, so the case asserts what a CUDA build DOES answer: measured on thor:gpu0 (sm_110, CUDA 13.0.88), rc job 1b46515d-8caf-4c49-823e-efc7a1f3e3f4, the request travels the whole registered forward and stops at the MoE router's named vision-bias refusal (an image step routes on exp_probs_b_vl, and the device router takes one bias per call with no per-row selector; that device arm is owed by #2411 W7-CUDA).

IT ASSERTS THE REFUSAL, NOT MERELY 'NOT W7-device'. A bare inequality would accept ANY failure, including a regression that stopped the request earlier - which this row has already lived through twice (the vision-residency refusal, then the host GEMM's wq_a layer 0). When W7-CUDA lands the per-row bias, the request stops failing and takes the 'served' branch instead.
