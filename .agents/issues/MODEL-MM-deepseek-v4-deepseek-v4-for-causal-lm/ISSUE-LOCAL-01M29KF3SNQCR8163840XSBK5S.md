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

-
