ID: ISSUE-LOCAL-01M2BRWKNZM8031JT1QE67WYHZ
Title: qwen4_exp loads on gfx1151 but the forward blocks in the AMDKFD SVM path (svm_range_set_attr, D state)
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

W6, the first load-and-forward of Qwen3.8-Flash-Next UD-IQ1_S on strix:gpu0 (gfx1151), LOADS and does NOT refuse any op, and the forward never completes. Measured 2026-09-12 at origin/main cfca943f6 under three rc leases.

WHAT WORKS. The 67.56 GiB UD-IQ1_S GGUF loads on ROCm: [vt load] weights 50.110 s, gguf prefault spans=534 paged_in=65.488 GiB in 37.107 s (1807.2 MiB/s), server ready 60 s after launch, VmHWM 27.67 GB. The forward then RUNS: 18 distinct ops announce on device 5 (kROCM), every one selected=vt-native priority=0 registered=1, so the reference-tier hit count is zero, which is half of the row's G2. In first-use order they are kEmbedding, kIndexSelect, kQwen4ExpGatedResidual, kMatmulBTQuant, kMatmulBT, kGdnStateGather, kCausalConv1dFwd, kGdnStateScatter, kGdnPostConv, kGdnPrefill, kRmsNormGated, kQwen4ExpGatedResidualWriteBack, kMoeRouterTopK, kMatmulBTQuantGrouped, kMoeSiluMul, kMatmul, kSharedExpertGate, kMoeCombine. No 'no kernel for op' line appears anywhere in any run, so W5 did remove the last op refusal and D2's refusal-by-name is NOT what stops this forward.

WHAT FAILS. No token is ever produced. Three runs: 900 s (curl exit 28, 0 bytes), 5 min sampled, 3 min sampled. It is BLOCKED, not slow: after the first sample the server process accumulates +11, +39, +14, +21 CPU ticks per 60 s (0.1-0.4 CPU-seconds) and /sys/class/drm/card*/device/gpu_busy_percent reads 0 throughout, and no new op ever announces.

WHERE IT BLOCKS. Sampled at t=180 s with AMD_SERIALIZE_KERNEL=3, the compute thread is in UNINTERRUPTIBLE sleep: tid state=D wchan=svm_range_set_attr syscall=16 (ioctl) on /dev/kfd. The remaining threads are 6 in futex_wait_queue, 2 in kfd_wait_on_events, 1 in pipe_read, 1 in inet_csk_accept. So the stall is inside the amdgpu/amdkfd SVM range path in the kernel driver, reached through an ioctl on /dev/kfd, and not in any vt:: kernel of ours.

LIKELY CAUSE, NOT YET PROVEN. The host side of strix:gpu0 is 30 GiB total / ~28 GiB available after the 96 GiB carve, and the model is 67.56 GiB. dmesg on the host carries 'amdgpu: SVM mapping failed, exceeds resident system memory limit' and 'User Buffer Address: 0x... already allocated by SVM'. Those dmesg lines are host-wide and their kernel timestamps CANNOT be pinned to this job, so they are corroborating and not conclusive. What is conclusive is the D state in svm_range_set_attr.

WHY THIS MATTERS BEYOND THE ROW. VT_ROCM_MANAGED_ALLOC was NOT set for any of these runs, and the thread is still in the SVM path, so on this board the ordinary allocation route reaches amdkfd SVM anyway. If that is right, the measured '27 GiB managed ceiling vs 76 GiB hipMalloc' split does not protect a large model here, and any gfx1151 model whose weights exceed host RAM may hit the same uninterruptible stall rather than an allocation error.

NEXT HYPOTHESIS. Run a model that fits host RAM (for example a small GGUF) through the same binary on the same board. If it forwards, the ceiling reading is confirmed and the gap is a device-residency question, not a kernel-correctness one. If it also stalls in svm_range_set_attr, the SVM path itself is the defect.

NO PERFORMANCE NUMBER IS ADMISSIBLE FROM THIS WORK. G3 first tokens is NOT met, G4 token-exactness is untouched, and the load timings above are recorded as load evidence only.

## Resolution

-
