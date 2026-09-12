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

2026-09-12 UPDATE, one of the two owed halves is DONE. The fallback arm now refuses by name: 'deepseek-v4 host GEMM: weight size mismatch: tensor `<name>` layer <n> want [N=..,K=..] = .. elements, got .. elements', in the sibling keep-quant arm's own vocabulary, and that arm was given the same tensor/layer identifiers so both name the same thing. The labels are REQUIRED rather than defaulted, because a defaulted label leaves a call site anonymous - the defect itself - so a forgotten site is a -Werror failure; check-tree-compiles compiled 685 of 685 translation units in scope. A red-first case in test_deepseek_v4_forward enters through the production entry DeepseekV4ForwardHost (Gemm and MatVec are file-local and no test can construct either) and asserts the tensor, the layer, both geometries and the actual element count, including that breaking layer 2 names LAYER 2 and not layer 0.

THIS ISSUE STAYS OPEN. Naming the refusal is the instrument, not the repair: the wrong shape is still thrown and a served image still does not complete. The N and K values, the tensor and the layer REMAIN UNMEASURED and are still not guessed here - they are recorded only once a lease has printed them.

## Resolution

-
