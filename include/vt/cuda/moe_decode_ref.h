// vllm.cpp — TEST-ONLY serial reference entry points for the parallelized MoE
// decode kernels (router top-k and moe_align_block_size). These launch the
// original single-block/single-thread reference paths so the CUDA parity tests
// can assert the parallel production kernels are byte-identical to them. Not on
// any production path. See src/vt/cuda/cuda_moe.cu and cuda_marlin_repack.cu.
#pragma once

#include <cstdint>

#include "vt/ops.h"

namespace vt::cuda {

// Serial-greedy top-k reference (single-threaded argmax). Same signature and
// semantics as the registered CUDA moe_router_topk, but forces the reference
// path. Used only by tests.
void MoeRouterTopKSerialCuda(Queue& q, Tensor& weights, Tensor& indices, const Tensor& logits,
                             const MoeRouterTopKArgs& args);

}  // namespace vt::cuda
