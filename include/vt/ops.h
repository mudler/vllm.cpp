// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include "vt/tensor.h"

namespace vt {

enum class OpId : uint8_t { kMatmul, kRmsNorm, kSiluAndMul, kRopeNeox, kEmbedding, kCount };

void RegisterOp(OpId op, DeviceType device, void* fn);
void* GetOp(OpId op, DeviceType device);

// out[M,N] = a[M,K] @ b[K,N]; a/b float dtypes (f32/f16/bf16), out f32,
// f32 accumulation, all contiguous, same device.
void Matmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);

}  // namespace vt
