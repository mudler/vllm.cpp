ID: ISSUE-LOCAL-01M2AYNBBZPVSG7G9PSGSGF68V
Title: include/vt/ops.h states five device-registration claims the tree falsifies
Row: MODEL-MM-QWEN4-EXP
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

Five OpId comment blocks in include/vt/ops.h claim a registration set that the tree contradicts, in the same shape as the two blocks MODEL-MM-QWEN4-EXP W3 staled. Each claim was checked against the RegisterOp call sites: kAllReduce/kAllGather (:281) name an MLX-ring kMETAL transport, and no kMETAL registration exists for any of the four collective ids; kMoeGroupedGemmBf16GateUpSilu (:353) says CUDA-only while src/vt/rocm/rocm_ops.hip:295 registers a kROCM arm, as :292 does for kMoeGroupedGemmBf16; kRmsNormGroup's rejection argument (:668) says kRmsNorm is registered on five backends when six register it (kCPU, kCUDA, kROCM, kVULKAN, kTENSTORRENT, kMETAL); kEmbeddingQuant (:773) lists ROCM among the unregistered arms although rocm_ops.hip:207 registers it; kGlm5NextKpoolCompress/kGlm5NextKpoolSelect (:827) say kCUDA ONLY although rocm_ops.hip:345,348 register kROCM arms. A comment that names the backends an op refuses is the document a caller reads before it writes a dispatch, so a wrong one routes work at a backend that would refuse it, or hides one that would serve it.

## Resolution

Fixed in the same flow as the W3 re-review repair (AGENTS.md, 'Every change starts from an issue': file it, fix it in the same flow, reference it in the commit, close it). All five blocks in include/vt/ops.h now state the registration set the RegisterOp call sites actually hold, each with the file:line that proves it: :281 collectives (kCPU communicator.cpp:297-302 and kCUDA nccl_communicator.cu:152-158 only, kMETAL MLX-ring OWED), :353 kMoeGroupedGemmBf16GateUpSilu (kCUDA cuda_matmul_nvfp4.cu:2725 + kROCM rocm_ops.hip:295), :668 kRmsNorm (six backends, named), :773 kEmbeddingQuant (kROCM rocm_ops.hip:207 moved out of the owed list), :827 GLM-5.3-Flash k-pool pair (kCUDA cuda_glm5_next.cu:561,564 + kROCM rocm_ops.hip:345,348). Comment-only: no object code changes, so no device re-gate is owed. Verified 2026-09-12 by grepping every RegisterOp(OpId::<id>, ...) site for each id.
