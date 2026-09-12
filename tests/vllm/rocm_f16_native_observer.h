// Test-only link interposition preserves the actual native provider selection.
// RegisterOp cannot replace a provider: duplicate names keep their first entry.
#pragma once

#include "vt/ops.h"

namespace rocm_f16_test {
inline vt::MatmulFn on_nn = nullptr;
inline vt::MatmulFn on_bt = nullptr;
inline vt::EmbeddingFn on_embedding = nullptr;
inline vt::CastF32Fn on_cast_f32 = nullptr;
extern "C" void RealNn(vt::Queue&, vt::Tensor&, const vt::Tensor&, const vt::Tensor&)
    asm("__real__ZN2vt4rocm16MatmulKernelRocmERNS_5QueueERNS_6TensorERKS3_S6_");
extern "C" void RealBt(vt::Queue&, vt::Tensor&, const vt::Tensor&, const vt::Tensor&)
    asm("__real__ZN2vt4rocm18MatmulBTKernelRocmERNS_5QueueERNS_6TensorERKS3_S6_");
extern "C" void RealEmbedding(vt::Queue&, vt::Tensor&, const vt::Tensor&, const vt::Tensor&)
    asm("__real__ZN2vt4rocm19EmbeddingKernelRocmERNS_5QueueERNS_6TensorERKS3_S6_");
extern "C" void RealCastF32(vt::Queue&, vt::Tensor&, const vt::Tensor&)
    asm("__real__ZN2vt4rocm17CastF32KernelRocmERNS_5QueueERNS_6TensorERKS3_");
extern "C" void WrapNn(vt::Queue&, vt::Tensor&, const vt::Tensor&, const vt::Tensor&)
    asm("__wrap__ZN2vt4rocm16MatmulKernelRocmERNS_5QueueERNS_6TensorERKS3_S6_");
extern "C" void WrapBt(vt::Queue&, vt::Tensor&, const vt::Tensor&, const vt::Tensor&)
    asm("__wrap__ZN2vt4rocm18MatmulBTKernelRocmERNS_5QueueERNS_6TensorERKS3_S6_");
extern "C" void WrapEmbedding(vt::Queue&, vt::Tensor&, const vt::Tensor&, const vt::Tensor&)
    asm("__wrap__ZN2vt4rocm19EmbeddingKernelRocmERNS_5QueueERNS_6TensorERKS3_S6_");
extern "C" void WrapCastF32(vt::Queue&, vt::Tensor&, const vt::Tensor&)
    asm("__wrap__ZN2vt4rocm17CastF32KernelRocmERNS_5QueueERNS_6TensorERKS3_");
extern "C" void WrapNn(vt::Queue& q, vt::Tensor& out, const vt::Tensor& a, const vt::Tensor& b) {
  (on_nn ? on_nn : RealNn)(q, out, a, b);
}
extern "C" void WrapBt(vt::Queue& q, vt::Tensor& out, const vt::Tensor& a, const vt::Tensor& b) {
  (on_bt ? on_bt : RealBt)(q, out, a, b);
}
extern "C" void WrapEmbedding(vt::Queue& q, vt::Tensor& out, const vt::Tensor& table, const vt::Tensor& ids) {
  (on_embedding ? on_embedding : RealEmbedding)(q, out, table, ids);
}
extern "C" void WrapCastF32(vt::Queue& q, vt::Tensor& out, const vt::Tensor& in) {
  (on_cast_f32 ? on_cast_f32 : RealCastF32)(q, out, in);
}
}  // namespace rocm_f16_test
