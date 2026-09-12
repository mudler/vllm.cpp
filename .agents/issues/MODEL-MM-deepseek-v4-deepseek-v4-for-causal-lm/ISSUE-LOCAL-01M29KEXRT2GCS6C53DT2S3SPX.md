ID: ISSUE-LOCAL-01M29KEXRT2GCS6C53DT2S3SPX
Title: DeepSeek-V4 vision: a served image on a CUDA build dies in an anonymous 'vt: MatVec weight size mismatch'
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

W7-CUDA staged the vision tower to the queue's device, so 'DeepSeek-V4 vision queue and weights must share one device' no longer blocks a served image on CUDA. The blocker MOVED rather than closed: the request now throws 'vt: MatVec weight size mismatch' from deepseek_v4.cpp:504, the UNCONDITIONAL VT_CHECK in Gemm's host MatVec arm. That guard is ANONYMOUS - it names no tensor, no layer, no geometry and nothing missing - while the sibling keep-quant arm refuses by name. The failing code is HOST code; what is device-specific is only its reachability, because ForwardDevice refuses earlier on a CPU build at VT_CHECK(V4DeviceKernelsAvailable(), kDevicePending). The N and K values, the tensor and the layer are UNMEASURED and are not guessed here: recovering them needs an instrumented device run under an rc lease. It is not the aarch64 repack path - the failure is byte-identical with VT_CPU_QUANT_REPACK=0. Owed: root-cause it, and give the fallback arm a named refusal. Measured by W7-CUDA on thor:gpu0, see .agents/specs/deepseek-v4-flash-vision.md section 'W7-CUDA evidence'.

## Resolution

-
