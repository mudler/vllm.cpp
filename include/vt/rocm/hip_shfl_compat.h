#ifndef VT_ROCM_HIP_SHFL_COMPAT_H
#define VT_ROCM_HIP_SHFL_COMPAT_H

// ROCm 5.7 / clang-17 does not define the CUDA-compatibility _sync wrappers.
// The native HIP API is __shfl_down(val, offset) / __shfl(val, lane) with no
// mask. Guard with #ifndef so newer ROCm (which defines these) uses its own.
#ifndef __shfl_down_sync
#define __shfl_down_sync(mask, val, offset) __shfl_down(val, offset)
#endif
#ifndef __shfl_sync
#define __shfl_sync(mask, val, lane) __shfl(val, lane)
#endif

#endif  // VT_ROCM_HIP_SHFL_COMPAT_H
