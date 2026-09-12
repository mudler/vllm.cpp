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

2026-09-12 MEASURED, on thor:gpu0 (sm_110, CUDA 13.0.88, aarch64), rc job 04f39bcb-5636-4043-9cfc-8bdebd4862ec. The served image reported: 'vt: deepseek-v4 host GEMM: weight size mismatch: tensor `wq_a` layer 0 want [N=32,K=32] = 1024 elements, got 0 elements'. So the tensor is wq_a, the layer is 0, and N=K=32 is the fixture's q_lora_rank by hidden_size.

ROOT CAUSE. 'got 0' is what rules the shape hypothesis out: a wrong shape gives a wrong COUNT, while zero means the weight reached NEITHER arm. DeepseekV4Model::ForwardDevice built its backend with gguf=nullptr and bound only the EXL3 tower, so ForwardComposeImpl read kq_src = be.gguf != nullptr as false and handed every layer Lq = nullptr; every Gemm then fell to the host-float arm, whose MLA tower is empty by design on a GGUF load (deepseek_v4_weights.cpp asserts hl.wq_a.empty()) because the weight is meant to be consumed keep-quant. Layer 0's wq_a is just the first GEMM the composition performs. The sibling DeepseekV4Model::Forward has always dispatched on has_gguf_weights; this entry never did (git log -S'dev_be.gguf' finds nothing since d7d1ee914), the registry routes the default gather_logits path here unconditionally (deepseek_v4_registry.cpp:246), and no test has ever driven this entry with a GGUF tower - so the combination that serving a real checkpoint uses was never executed.

REPAIR COMMITTED on this row's branch: ForwardDevice binds the keep-quant tower when the load took that arm. It binds rather than delegating to DeepseekV4ForwardGguf, because the four device op families are the point of that entry; binding gguf sets dsa_dense, which turns the indexer and compressor arms off on every layer exactly as the GGUF sibling already does.

THIS ISSUE STAYS OPEN until a lease records what the served image does with the tower bound. Whether an image COMPLETES is a separate measurement and is not inferred from the diagnosis - it may yet stop at a further blocker, which would be named rather than guessed.

## Resolution

-
