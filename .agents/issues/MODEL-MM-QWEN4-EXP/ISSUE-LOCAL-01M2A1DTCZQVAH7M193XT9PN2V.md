ID: ISSUE-LOCAL-01M2A1DTCZQVAH7M193XT9PN2V
Title: qwen4_exp cannot load or decode on ROCm: nine vt ops have no arm and hard-refuse
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

Nine vt operations the qwen4_exp forward calls have no ROCm arm, so the model cannot execute a forward pass on any AMD device. Measured on the tree at 97cb6964b by enumerating the ops the model TUs call and checking RegisterOp for each.

No ROCm arm (registered for kCPU and kCUDA only):

  kQwen4ExpPleConv                  cpu_qwen4_exp_ple.cpp:247   cuda_qwen4_exp_ple.cu:371
  kQwen4ExpPleGate                  cpu_qwen4_exp_ple.cpp:250   cuda_qwen4_exp_ple.cu:374
  kQwen4ExpGatedResidual            cpu_qwen4_exp.cpp:317       cuda_qwen4_exp.cu:447
  kQwen4ExpGatedResidualWriteBack   cpu_qwen4_exp.cpp:320       cuda_qwen4_exp.cu:450
  kQwen4ExpQsaCompress              cpu_qwen4_exp_qsa.cpp:354   cuda_qwen4_exp_qsa.cu:641
  kQwen4ExpQsaGatherAttention       cpu_qwen4_exp_qsa.cpp:357   cuda_qwen4_exp_qsa.cu:644
  kRmsNormGroup                     cpu_ops.cpp:4237            cuda_rms_norm_group.cu:210
  kIndexSelect                      cpu_ops.cpp                 cuda_gdn.cu
  kIndexCopy                        cpu_ops.cpp                 cuda_gdn.cu

grep -rn 'Qwen4Exp' src/vt/rocm/ returns nothing.

THESE HARD-REFUSE ON gfx1151; THEY DO NOT FALL BACK. That is the load-bearing fact and it is counter-intuitive, because Strix Halo is an APU and the tree does carry a portable CPU reference tier. The chain, verified rather than inferred:

- ReferenceTierEligible (src/vt/op_provider.cpp:906) gates on DeviceMemoryIsHostAddressable(), deliberately NOT on UnifiedMemory(); that narrowing cost two crashes, #844 and #1435.
- RocmBackend::DeviceMemoryIsHostAddressable() returns unified_memory_ (rocm_backend.hip:484).
- ResolveMemoryPolicy (include/vt/rocm/rocm_arch.h:192) computes unified_memory = managed_alloc || (pageable_memory_access && integrated).
- gfx1151 reports pageableMemoryAccess=0, measured on strix:gpu0 under an rc lease, and since the #2511 narrowing managed_alloc = pageable_memory_access under kUnset.

So unified_memory_ is false, the tier is never installed, and GetOp refuses by name. rocm_backend.hip:313 says 'a discrete AMD board never installs the tier'; post-#2511 this APU behaves like one for that purpose. The repair that stopped the gfx1151 GPU hang also removed the CPU fallback on that board, and nothing records that coupling. Some spec prose still asserts host-addressability is true on gfx1151, which is pre-#2511 and now stale.

Why this is now the whole remaining gap for this model on AMD. The other two blockers are gone: #3097 landed the ROCm quantized gather, so DeviceQuantGatherSupported(kROCM) is true and the loader no longer refuses the device by name; and the IQ4_NL keep-quant GEMM gives the 48 ffn_down_exps a device arm. The weights are therefore loadable and multipliable on ROCm and the forward still cannot run.

The artifact fits, which is what makes this worth doing now rather than later. strix:gpu0 was re-carved to 96 GiB on 2026-09-11 and plain hipMalloc reaches at least 76 GiB there, measured by a bounded probe; the released unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S is 67.56 GiB. Note hipMallocManaged on that board tops out at host RAM, 27 GiB, so VT_ROCM_MANAGED_ALLOC=1 must not be set; the default already selects plain hipMalloc.

Each op has BOTH a CPU and a CUDA implementation to port from, about 1709 lines of CUDA across cuda_qwen4_exp.cu, cuda_qwen4_exp_ple.cu, cuda_qwen4_exp_qsa.cu and cuda_rms_norm_group.cu, and the CUDA sources use only __shfl_down_sync, atomicAdd and bf16/fp16 types. No __dp4a and no CUDA-only primitive appears in them, so the port is bounded rather than open-ended. vLLM at the current pin e126687a9a additionally ships a first-party AMD backend for this architecture, vllm/models/qwen4_exp/amd/, whose divergence from nvidia/ is about 840 lines confined to the Triton op layer with model.py, model_state.py and mtp.py byte-identical, and it carries Triton kernels for exactly these components. That is a mirror source for the AMD arm rather than a transliteration of the CUDA one.

## Resolution

-
