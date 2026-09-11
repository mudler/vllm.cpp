ID: ISSUE-GH-201
Title: rocm_matmul_hipblaslt.hip:384:13: error: no matching function for call to 'hipblasGemmEx'
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: 201
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-09
Updated: 2026-08-10
Closed: 2026-08-10

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ```
> [ 34%] Building HIP object CMakeFiles/vllm.dir/src/vt/rocm/rocm_gemma4_expert_geglu.hip.o
> /mnt/storage/llm/vllm.cpp/src/vt/rocm/rocm_matmul_hipblaslt.hip:384:13: error: no matching function for call to 'hipblasGemmEx'
>   384 |   CheckBlas(hipblasGemmEx(ctx.handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
>       |             ^~~~~~~~~~~~~
> /usr/include/hipblas/hipblas.h:31114:32: note: candidate function not viable: no known conversion from 'const hipDataType' to 'hipblasDatatype_t' for 9th argument
>  31114 | HIPBLAS_EXPORT hipblasStatus_t hipblasGemmEx(hipblasHandle_t    handle,
>        |                                ^
>  31115 |                                              hipblasOperation_t transA,
>  31116 |                                              hipblasOperation_t transB,
>  31117 |                                              int                m,
>  31118 |                                              int                n,
>  31119 |                                              int                k,
>  31120 |                                              const void*        alpha,
>  31121 |                                              const void*        A,
>  31122 |                                              hipblasDatatype_t  aType,
>        |                                              ~~~~~~~~~~~~~~~~~~~~~~~~
> /mnt/storage/llm/vllm.cpp/src/vt/rocm/rocm_matmul_hipblaslt.hip:442:13: error: no matching function for call to 'hipblasGemmEx'
>   442 |   CheckBlas(hipblasGemmEx(ctx.handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
>       |             ^~~~~~~~~~~~~
> /usr/include/hipblas/hipblas.h:31114:32: note: candidate function not viable: no known conversion from 'const hipDataType' to 'hipblasDatatype_t' for 9th argument
>  31114 | HIPBLAS_EXPORT hipblasStatus_t hipblasGemmEx(hipblasHandle_t    handle,
>        |                                ^
>  31115 |                                              hipblasOperation_t transA,
>  31116 |                                              hipblasOperation_t transB,
>  31117 |                                              int                m,
>  31118 |                                              int                n,
>  31119 |                                              int                k,
>  31120 |                                              const void*        alpha,
>  31121 |                                              const void*        A,
>  31122 |                                              hipblasDatatype_t  aType,
>        |                                              ~~~~~~~~~~~~~~~~~~~~~~~~
> /mnt/storage/llm/vllm.cpp/src/vt/rocm/rocm_matmul_hipblaslt.hip:479:13: error: no matching function for call to 'hipblasGemmEx'
>   479 |   CheckBlas(hipblasGemmEx(ctx.handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
>       |             ^~~~~~~~~~~~~
> /usr/include/hipblas/hipblas.h:31114:32: note: candidate function not viable: no known conversion from 'const hipDataType' to 'hipblasDatatype_t' for 9th argument
>  31114 | HIPBLAS_EXPORT hipblasStatus_t hipblasGemmEx(hipblasHandle_t    handle,
>        |                                ^
>  31115 |                                              hipblasOperation_t transA,
>  31116 |                                              hipblasOperation_t transB,
>  31117 |                                              int                m,
>  31118 |                                              int                n,
>  31119 |                                              int                k,
>  31120 |                                              const void*        alpha,
>  31121 |                                              const void*        A,
>  31122 |                                              hipblasDatatype_t  aType,
>        |                                              ~~~~~~~~~~~~~~~~~~~~~~~~
> /mnt/storage/llm/vllm.cpp/src/vt/rocm/rocm_matmul_hipblaslt.hip:509:13: error: no matching function for call to 'hipblasGemmStridedBatchedEx'
>   509 |   CheckBlas(hipblasGemmStridedBatchedEx(
>       |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~
> /usr/include/hipblas/hipblas.h:31869:32: note: candidate function not viable: no known conversion from 'const hipDataType' to 'hipblasDatatype_t' for 9th argument
>  31869 | HIPBLAS_EXPORT hipblasStatus_t hipblasGemmStridedBatchedEx(hipblasHandle_t    handle,
>        |                                ^
>  31870 |                                                            hipblasOperation_t transA,
>  31871 |                                                            hipblasOperation_t transB,
>  31872 |                                                            int                m,
>  31873 |                                                            int                n,
>  31874 |                                                            int                k,
>  31875 |                                                            const void*        alpha,
>  31876 |                                                            const void*        A,
>  31877 |                                                            hipblasDatatype_t  aType,
>        |                                                            ~~~~~~~~~~~~~~~~~~~~~~~~
> /mnt/storage/llm/vllm.cpp/src/vt/rocm/rocm_matmul_hipblaslt.hip:569:13: error: no matching function for call to 'hipblasGemmBatchedEx'
>   569 |   CheckBlas(hipblasGemmBatchedEx(
>       |             ^~~~~~~~~~~~~~~~~~~~
> /usr/include/hipblas/hipblas.h:31476:32: note: candidate function not viable: no known conversion from 'const hipDataType' to 'hipblasDatatype_t' for 9th argument
>  31476 | HIPBLAS_EXPORT hipblasStatus_t hipblasGemmBatchedEx(hipblasHandle_t    handle,
>        |                                ^
>  31477 |                                                     hipblasOperation_t transA,
>  31478 |                                                     hipblasOperation_t transB,
>  31479 |                                                     int                m,
>  31480 |                                                     int                n,
>  31481 |                                                     int                k,
>  31482 |                                                     const void*        alpha,
>  31483 |                                                     const void*        A[],
>  31484 |                                                     hipblasDatatype_t  aType,
>        |                                                     ~~~~~~~~~~~~~~~~~~~~~~~~
> /mnt/storage/llm/vllm.cpp/src/vt/rocm/rocm_matmul_hipblaslt.hip:628:13: error: no matching function for call to 'hipblasGemmBatchedEx'
>   628 |   CheckBlas(hipblasGemmBatchedEx(
>       |             ^~~~~~~~~~~~~~~~~~~~
> /usr/include/hipblas/hipblas.h:31476:32: note: candidate function not viable: no known conversion from 'const hipDataType' to 'hipblasDatatype_t' for 9th argument
>  31476 | HIPBLAS_EXPORT hipblasStatus_t hipblasGemmBatchedEx(hipblasHandle_t    handle,
>        |                                ^
>  31477 |                                                     hipblasOperation_t transA,
>  31478 |                                                     hipblasOperation_t transB,
>  31479 |                                                     int                m,
>  31480 |                                                     int                n,
>  31481 |                                                     int                k,
>  31482 |                                                     const void*        alpha,
>  31483 |                                                     const void*        A[],
>  31484 |                                                     hipblasDatatype_t  aType,
>        |                                                     ~~~~~~~~~~~~~~~~~~~~~~~~
> [ 34%] Building HIP object CMakeFiles/vllm.dir/src/vt/rocm/rocm_fp8_channel_gemv.hip.o
> 6 errors generated when compiling for gfx1100.
> gmake[2]: *** [CMakeFiles/vllm.dir/build.make:5198: CMakeFiles/vllm.dir/src/vt/rocm/rocm_matmul_hipblaslt.hip.o] Error 1
> gmake[2]: *** Waiting for unfinished jobs....
> gmake[1]: *** [CMakeFiles/Makefile2:1302: CMakeFiles/vllm.dir/all] Error 2
> gmake: *** [Makefile:146: all] Error 2
> ```
>
>     Root cause: The code used hipDataType (from hipBLASLt headers, constants like HIP_R_32F) but the actual hipblasGemmEx / hipblasGemmBatchedEx / hipblasGemmStridedBatchedEx API signatures take hipblasDatatype_t. On ROCm 6.4 these are distinct enum types with no implicit conversion.
>
> git rev-parse HEAD
> 81291a89f336928819245766ef44a8db27b42c00

## Resolution

Commit `fb6bb8b32e2aadbc61a961e53727f239283eccd9` dated 2026-08-09 selects the ROCm 6.x hipBLAS API generation and names issue #201. GitHub closed issue #201 on 2026-08-10.
