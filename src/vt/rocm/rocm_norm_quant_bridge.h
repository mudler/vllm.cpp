// Lever C (GFX1100-TG200-NORMQ): host-side bridge for the producer-fused Q8_K
// norm epilogue. Implemented in src/vt/rocm/rocm_grouped_gemm.hip (which owns
// the activation-quant scratch pool and the standalone QuantizeQ8KK launch);
// consumed by src/vt/rocm/rocm_rmsnorm.hip (the producer side).
//
// Contract (VT_NORM_QUANT_FUSED=1, opt-in; default OFF leaves every path
// byte-unchanged):
//   1. A producer dispatching an epilogue-enabled RmsNormRowKernel allocates
//      Q8_K scratch from the SAME grow-only stream-ordered pool the consumer
//      uses, launches the kernel with the epilogue pointer, and RECORDS a
//      single-slot token {out ptr, rows, h, dtype, scratch, stream}.
//   2. A MatmulBTQuant K-quant dispatch whose activation EXACTLY matches the
//      recorded token (same device pointer, rows, row length, stride, input
//      dtype) SKIPS its standalone QuantizeQ8KK launch and consumes the
//      produced scratch. The token survives matching consumers (the model's
//      attn q/k/v matvecs re-quantize ONE normalized row three times) and is
//      INVALIDATED by any non-matching K-quant consumer, so a stale token can
//      never serve a different buffer.
//   Stream-ordering argument: producer and consumer are enqueued on one
//   stream, and the epilogue quantizes the same global bf16 rows the
//   standalone kernel would read, through the SAME shared QuantQ8KSBlock body
//   -- byte equality holds by construction (asserted op-level in
//   tests/vt/test_rocm_quant_dot.cpp). Under hipGraph capture both sides run
//   at capture time, so the baked graph references the retired-never scratch
//   pointer exactly like the pre-existing pool discipline.
#ifndef VLLM_CPP_SRC_VT_ROCM_ROCM_NORM_QUANT_BRIDGE_H_
#define VLLM_CPP_SRC_VT_ROCM_ROCM_NORM_QUANT_BRIDGE_H_

#include <cstdint>

#include "vt/dtype.h"

namespace vt::rocm {

// Producer side: scratch of `bytes` from the quant pool on `s`, then record.
void* NormQuantProducerScratch(size_t bytes, void* stream);
void NormQuantRecordProducer(const void* out_ptr, int64_t rows, int64_t h,
                             int64_t row_stride, DType adt, const void* scratch,
                             void* stream);
// Consumer side: true + scratch when the activation matches the live token;
// false otherwise (and any non-matching query invalidates the token).
bool NormQuantTakeConsumer(const void* a_ptr, int64_t rows, int64_t h,
                           int64_t row_stride, DType adt, void* stream,
                           const void** scratch_out);

struct NormQuantCounts {
  long long producers;            // epilogue-enabled RmsNorm dispatches
  long long consumers_fused;      // K-quant dispatches that skipped the standalone quant
  long long consumers_standalone; // K-quant dispatches that launched QuantizeQ8KK
};
NormQuantCounts NormQuantCountsForTesting();
void NormQuantResetForTesting();
const void* NormQuantLastScratchForTesting();

}  // namespace vt::rocm

#endif  // VLLM_CPP_SRC_VT_ROCM_ROCM_NORM_QUANT_BRIDGE_H_
