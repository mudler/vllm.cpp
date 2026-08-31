// The CUDA dequantizing-gather seam (KGATHER).
//
// `EmbeddingKernelCuda` (cuda_ops.cu) owns the gather's error ring and its
// float path; the BLOCK path lives in cuda_quant_dot.cu, which is the only
// translation unit that defines the device codebooks
// (`cuda_quant_iq_tables.cuh` defines them, so a second includer is a duplicate
// symbol). This header is the two-function seam between them, plus the error
// record both sides write -- stated ONCE here rather than mirrored, because a
// layout that drifted between the writer and the reader would report a wrong id
// and never fail to compile.
#pragma once

#include <cuda_runtime.h>

#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vt::cuda {

// The device-side out-of-range report. `status` is the atomicCAS latch, `id`
// the first offending token id. `pad` keeps `id` naturally aligned and is why
// this is a struct rather than two pointers.
struct EmbeddingQuantErr {
  int status;    // 0 = ok, 1 = bad id recorded
  int pad;       // keep `id` naturally aligned
  long long id;  // first out-of-range id seen (valid when status != 0)
};

// True when this build has a DEVICE row decoder for `dt`. The list is the same
// one `vt::cpu::BlockToFloat` answers for, so a table the loader can keep on the
// CPU can also be kept on CUDA -- `tests/vt/test_cuda_embedding_quant.cpp` pins
// that equality, and `DeviceQuantGatherSupported` in the GGUF residency policy
// depends on it holding.
bool EmbeddingQuantSupported(DType dt);

// Gather `ids` out of a block-quantized `table` into `out`, decoding one row per
// id. Preconditions are the ones `vt::Embedding` already checked: rank-2 table,
// contiguous everything, `table.shape[1] % BlockElems(dtype) == 0`, out f32 or
// bf16, ids i32 or i64. Returns the launch status; an out-of-range id is
// clamped in-kernel and recorded in `err` rather than throwing, exactly as the
// float path does.
cudaError_t LaunchEmbeddingQuant(cudaStream_t s, Tensor& out, const Tensor& table,
                                 const Tensor& ids, EmbeddingQuantErr* err);

}  // namespace vt::cuda
