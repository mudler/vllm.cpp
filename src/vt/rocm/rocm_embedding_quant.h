// Native packed-row gather seam for BACKEND-ROCM-QUANT-GATHER (#3093).
#pragma once

#include <cstdint>

#include "vt/ops.h"

namespace vt::rocm {

// Constant-size scratch copied after the gather finishes on its stream.
struct EmbeddingQuantError {
  int32_t status;
  int32_t reserved;
  int64_t id;
};
static_assert(sizeof(EmbeddingQuantError) == 16);

bool EmbeddingQuantSupported(DType dtype);
void EmbeddingQuantKernelRocm(Queue& queue, Tensor& out, const Tensor& table,
                              const Tensor& ids);

}  // namespace vt::rocm
