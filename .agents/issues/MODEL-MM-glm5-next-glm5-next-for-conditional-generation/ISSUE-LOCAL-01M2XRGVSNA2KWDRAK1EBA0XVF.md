ID: ISSUE-LOCAL-01M2XRGVSNA2KWDRAK1EBA0XVF
Title: Wire GLM-5.3-Flash mHC pre/post to device kernels (kDeepseekV4Mhc)
Row: MODEL-MM-glm5-next-glm5-next-for-conditional-generation
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-19
Updated: 2026-09-19
Closed: -

## Problem

The GLM-5.3-Flash device forward (Glm5NextDeviceForward) runs mHC pre/post as host-fallback islands on the interposed CPU queue. The kDeepseekV4Mhc op has both CUDA (O34) and ROCm providers registered, and MhcDevice()->pre()/post() share the same host-vector interface as V4. The device forward should probe for the op and route pre/post through device kernels when available, falling back to host MhcPre/MhcPost on CPU. HcHeadCollapseMean stays on host: GLM-5.3 uses an unweighted mean while V4's weighted head kernel diverges.

## Resolution

Implementation committed on `row/MODEL-MM-GLM53-FLASH-W9C-MHC-DEV` (PR #3229):
- `glm5_next_device.cpp`: added `#include` for `deepseek_v4_device.h`,
  added `mhc_on_device` probe using `vt::OpRegistered(kDeepseekV4Mhc, ...)`,
  wired all 4 mHC sites (attn pre/post, ffn pre/post) to call
  `MhcDevice()->pre()/post()` when the op is registered on the queue's
  device, falling back to host `MhcPre`/`MhcPost` on CPU.
- `.agents/specs/glm5-next-flash.md`: arm table row 11 updated from
  HOST ISLAND to ON DEVICE, O34 owed-item text and Outcome section
  updated to reflect ten of eleven arms on device.

Gate on thor:gpu0 (CPU-only build, no CUDA compiler):
- Default (no device flag): 33/33 passed.
- VT_GLM5_NEXT_DEVICE=1: 22 passed, 11 failed (pre-existing #2348).
- No new failures introduced. The mHC device kernel path is not
  exercised on a CPU-only build (host fallback used), same as before.
