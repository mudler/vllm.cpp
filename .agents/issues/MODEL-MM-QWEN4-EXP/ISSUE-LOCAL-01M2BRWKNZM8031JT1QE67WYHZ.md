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

W7 ANSWERS THE NEXT HYPOTHESIS: A MODEL THAT FITS HOST RAM DOES FORWARD. Measured 2026-09-12 at origin/main 8ea148d4e on strix:gpu0 under three rc leases, box idle and exclusively leased, VT_ROCM_MANAGED_ALLOC UNSET throughout. Qwen3.8-27B-Q4_K_M.gguf (15.93 GiB, sha256 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169, architecture Qwen3_5ForConditionalGeneration) loads and FORWARDS TO COMPLETION through the same examples/vllm-server binary, on the same board, with --device auto: HTTP 200 and real text on 3 of 5 attempts, up to 64 completion tokens, peak VmHWM 23.98 GB of 30 GiB. Every announced op reads device=5 (kROCM) selected=vt-native, and the reference-tier hit count is 0, so the forward ran on the GPU and not on the portable CPU tier.

THE STALL IS SIZE-BOUND, NOT PATH-BOUND, AND THE A/B IS ON ONE BINARY. In the SAME lease, minutes apart, that same binary was pointed at the 67.56 GiB UD-IQ1_S rung and reproduced this issue exactly: tid state=D wchan=svm_range_set_attr syscall=16, held for 39 consecutive 20 s samples across the whole 805 s request, gpu_busy_percent 0-1, cpu_ticks moving by 1-2 per 20 s, and no token. The small rung's compute thread passes THROUGH the paging wait (wchan=folio_wait_bit_common, the CIFS page-in) and proceeds; the big rung's thread enters folio_wait_bit_common, transitions to svm_range_set_attr, and never leaves. So the SVM path is not categorically broken on gfx1151, and this is a device/host RESIDENCY problem: it belongs to MODEL-MM-QWEN4-EXP, and it wants a smaller rung or a placement change, not a BACKEND-ROCM rewrite.

WHAT W7 DOES NOT PROVE. The graph was NOT held constant. No qwen4_exp rung smaller than 67.56 GiB exists on /workspace (the only other qwen4_exp artifact is the BF16 at /workspace/q4exp-apex, which is larger), so the forwarding model is a different architecture. W7 therefore rules out "every model on this board stalls in the SVM path" and does not on its own acquit the qwen4_exp graph. It also cannot bisect the threshold: the largest GGUF that forwards is 15.93 GiB and the smallest that stalls is 67.56 GiB, with nothing in between staged, so the residency limit is bracketed only to that range.

W7 ALSO SHARPENS THE LOAD PICTURE. Under W7's leases the same 67.56 GiB artifact prefaulted at 92.2 MiB/s (65.488 GiB in 727.665 s, weights 787.678 s, ready after 794 s) against W6's 1807.2 MiB/s, because W6 read a warm CIFS cache and W7 read a cold one. Load wall time on this board is a property of the share, not of the loader, and neither figure gates anything.

G3 REMAINS UNMET FOR THIS ROW. The tokens above are another model's. No qwen4_exp token has been produced on gfx1151.

W7 ALSO FOUND A SEPARATE DEFECT, filed as its own local issue against BACKEND-ROCM (ISSUE-LOCAL-01M2BY2M2ATNVR3XQKV2DB1BJD): the forwarding 15.93 GiB rung fails 2 of 5 identical greedy-decode requests with an illegal GPU memory access ("Memory access fault by GPU node-1 ... Page not present" in one run, "[vt rocm: hipMemcpyAsync: an illegal memory access was encountered]" in another). That is a different failure from this one - loud, fast and intermittent, with no thread ever in svm_range_set_attr - but it means the gfx1151 arm is not clean even below the residency limit, and it bears on how much any future gate on this board can be trusted.

## Resolution

-
