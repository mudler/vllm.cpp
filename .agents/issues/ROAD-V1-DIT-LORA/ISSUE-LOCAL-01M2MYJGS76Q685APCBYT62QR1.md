ID: ISSUE-LOCAL-01M2MYJGS76Q685APCBYT62QR1
Title: cuda_gdn.cu fails to compile with CUDA 12.0: cudaGetDriverEntryPointByVersion undefined
Row: ROAD-V1-DIT-LORA
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-16
Updated: 2026-09-16
Closed: 2026-09-16

## Problem

src/vt/cuda/cuda_gdn.cu uses cudaGetDriverEntryPointByVersion (introduced in CUDA 12.2) without a version guard. On Thor (aarch64, CUDA 12.0 from nvidia-cuda-toolkit), compilation fails at line 231. cudaGetDriverEntryPoint (CUDA 12.0) is available and can be used as a fallback.

## Resolution

Added CUDART_VERSION < 12020 compat shim in src/vt/cuda/cuda_gdn.cu. Committed 623a0683c on row/DIT-LORA-GENERIC.
