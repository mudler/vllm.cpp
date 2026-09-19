ID: ISSUE-LOCAL-01M2XRGVSNA2KWDRAK1EBA0XVF
Title: Wire GLM-5.3-Flash mHC pre/post to device kernels (kDeepseekV4Mhc)
Row: MODEL-MM-glm5-next-glm5-next-for-conditional-generation
State: CLOSED
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-19
Updated: 2026-09-19
Closed: 2026-09-19

## Problem

The GLM-5.3-Flash device forward (Glm5NextDeviceForward) runs mHC pre/post as host-fallback islands on the interposed CPU queue. The kDeepseekV4Mhc op has both CUDA (O34) and ROCm providers registered, and MhcDevice()->pre()/post() share the same host-vector interface as V4. The device forward should probe for the op and route pre/post through device kernels when available, falling back to host MhcPre/MhcPost on CPU. HcHeadCollapseMean stays on host: GLM-5.3 uses an unweighted mean while V4's weighted head kernel diverges.

## Resolution

Work landed in PR #3229 (merged as 8f29d43b2). Ten of eleven arms on device; mHC pre/post now route through device kernels via kDeepseekV4Mhc probe with host fallback.
