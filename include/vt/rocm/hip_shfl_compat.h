#ifndef VT_ROCM_HIP_SHFL_COMPAT_H
#define VT_ROCM_HIP_SHFL_COMPAT_H

// ROCm <6 does not define the CUDA-compatibility _sync wrappers as functions.
// The native HIP API is __shfl_down(val, offset) / __shfl(val, lane) with no
// mask. On ROCm >=6 these are native inline templates in amd_warp_sync_functions.h,
// so defining them as macros here would intercept the function definitions in
// amd_hip_bf16.h. Use the HIP version, not #ifndef, because #ifndef cannot
// detect a function — only a macro.
#if HIP_VERSION_MAJOR < 6
#ifndef __shfl_down_sync
#define __shfl_down_sync(mask, val, offset) __shfl_down(val, offset)
#endif
#ifndef __shfl_sync
#define __shfl_sync(mask, val, lane) __shfl(val, lane)
#endif
#endif

#endif  // VT_ROCM_HIP_SHFL_COMPAT_H
