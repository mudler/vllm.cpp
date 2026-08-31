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

// THE RESIDENCY FLIP, as ONE callable.
//
// Since the GGUF gate became `OpRegistered(kEmbeddingQuant, dev)`, registering
// this device's block gather IS the flip. This function performs that
// registration and NOTHING else, so the flip is one call rather than a
// copy-pasted registrar line, and so the device gate can enable it in test scope
// while production still refuses.
//
// PRODUCTION DOES NOT CALL THIS YET. `cuda_ops.cu`'s registrar has the call
// commented out with its reason: nvcc has never compiled the decoders and no
// device has executed them, and registering early would route every GGUF model's
// block-typed gather table on CUDA into never-executed code with no throw and no
// log if the decode were wrong. `tests/vt/test_cuda_embedding_quant.cpp` calls it
// to measure the kernel through the real dispatch, and asserts that production
// has NOT called it. When that gate runs green on a CUDA host, the registrar
// calls this and that assertion becomes its opposite, in one commit.
void RegisterCudaBlockGather();

}  // namespace vt::cuda
