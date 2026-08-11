// ROCm op-table registration (BACKEND-ROCM).
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace vt::rocm {

void RmsNormKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                       const RmsNormArgs& args, Tensor* residual);
void EmbeddingKernelRocm(Queue& q, Tensor& out, const Tensor& table, const Tensor& ids);
void MulScalarKernelRocm(Queue& q, Tensor& out, const Tensor& x, double scalar);
void SoftCapKernelRocm(Queue& q, Tensor& out, const Tensor& x, double cap);
void AddKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);
void GeluAndMulKernelRocm(Queue& q, Tensor& out, const Tensor& x);
void SiluAndMulKernelRocm(Queue& q, Tensor& out, const Tensor& x);
void MatmulKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);
void MatmulBTKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);
void RopeCosSinCacheKernelRocm(Queue& q, Tensor& cos_sin, const Tensor& pos,
                               const RopeArgs& args);
void GreedyArgmaxKernelRocm(Queue& q, Tensor& out, const Tensor& logits);
void CastBf16KernelRocm(Queue& q, Tensor& out, const Tensor& in);
void CastF32KernelRocm(Queue& q, Tensor& out, const Tensor& in);
void QkvSplitKernelRocm(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out,
                        const Tensor& qkv);
void LayerNormKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor* weight,
                         const Tensor* bias, const LayerNormArgs& args);
void ReluKernelRocm(Queue& q, Tensor& out, const Tensor& x);
void RopeNeoxKernelRocm(Queue& q, Tensor& qs, Tensor& ks, const Tensor& pos,
                        const RopeArgs& args);
void RopeFromCacheKernelRocm(Queue& q, Tensor& qs, Tensor* ks, const Tensor& positions,
                             const Tensor& cache, const RopeArgs& args);
void ReshapeAndCacheKernelRocm(Queue& q, const Tensor& k, const Tensor& v, Tensor& k_cache,
                               Tensor& v_cache, const Tensor& slot_mapping);
void PagedAttentionKernelRocm(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                              const Tensor& v_cache, const Tensor& block_table,
                              const Tensor& seq_lens, const Tensor& query_start_loc,
                              const PagedAttentionArgs& args);
void GeluTanhKernelRocm(Queue& q, Tensor& out, const Tensor& x);
void GeluErfKernelRocm(Queue& q, Tensor& out, const Tensor& x);
void MoeRouterTopKKernelRocm(Queue& q, Tensor& weights, Tensor& indices, const Tensor& logits,
                             const MoeRouterTopKArgs& args, const Tensor* bias);
void ApplyAllowedTokenIdsKernelRocm(Queue& q, Tensor& logits, const Tensor& mask);
void ApplyTokenMaskKernelRocm(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols);
void ApplyTemperatureKernelRocm(Queue& q, Tensor& logits, const Tensor& temp, bool all_random);
void ApplyTopKTopPKernelRocm(Queue& q, Tensor& logits, const Tensor* k, const Tensor* p);
void ComputeProbsKernelRocm(Queue& q, Tensor& probs, const Tensor& logits);
void ComputeLogprobsKernelRocm(Queue& q, Tensor& logprobs, const Tensor& logits);
void RandomSampleKernelRocm(Queue& q, Tensor& token_ids, const Tensor& probs, const Tensor& seeds);
void ApplyPenaltiesKernelRocm(Queue& q, Tensor& logits, const Tensor& prompt_mask,
                              const Tensor& output_bin_counts, const Tensor& output_mask,
                              const Tensor& frequency_penalties, const Tensor& presence_penalties,
                              const Tensor& repetition_penalties);
void ApplyMinPKernelRocm(Queue& q, Tensor& logits, const Tensor& min_p);
void ApplyLogitBiasKernelRocm(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols,
                              const Tensor& biases);

namespace {

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kRmsNorm, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernelRocm)));
    RegisterOp(OpId::kEmbedding, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernelRocm)));
    RegisterOp(OpId::kMulScalar, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MulScalarFn>(&MulScalarKernelRocm)));
    RegisterOp(OpId::kSoftCap, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<SoftCapFn>(&SoftCapKernelRocm)));
    RegisterOp(OpId::kAdd, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<AddFn>(&AddKernelRocm)));
    RegisterOp(OpId::kGeluAndMul, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<GeluAndMulFn>(&GeluAndMulKernelRocm)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernelRocm)));
    RegisterOp(OpId::kMatmul, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernelRocm)));
    RegisterOp(OpId::kMatmulBT, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTKernelRocm)));
    RegisterOp(OpId::kRopeCosSinCache, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<RopeCosSinCacheFn>(&RopeCosSinCacheKernelRocm)));
    RegisterOp(OpId::kGreedyArgmax, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<GreedyArgmaxFn>(&GreedyArgmaxKernelRocm)));
    RegisterOp(OpId::kCastBf16, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastBf16KernelRocm)));
    RegisterOp(OpId::kCastF32, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<CastF32Fn>(&CastF32KernelRocm)));
    RegisterOp(OpId::kQkvSplit, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<QkvSplitFn>(&QkvSplitKernelRocm)));
    RegisterOp(OpId::kLayerNorm, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<LayerNormFn>(&LayerNormKernelRocm)));
    RegisterOp(OpId::kRelu, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ReluFn>(&ReluKernelRocm)));
    RegisterOp(OpId::kGeluTanh, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ReluFn>(&GeluTanhKernelRocm)));
    RegisterOp(OpId::kGeluErf, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ReluFn>(&GeluErfKernelRocm)));
    RegisterOp(OpId::kRopeNeox, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<RopeFn>(&RopeNeoxKernelRocm)));
    RegisterOp(OpId::kRopeFromCache, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<RopeFromCacheFn>(&RopeFromCacheKernelRocm)));
    RegisterOp(OpId::kReshapeAndCache, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernelRocm)));
    RegisterOp(OpId::kPagedAttention, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<PagedAttentionFn>(&PagedAttentionKernelRocm)));
    RegisterOp(OpId::kMoeRouterTopK, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<MoeRouterTopKFn>(&MoeRouterTopKKernelRocm)));
    // Full V1 sampler surface for Hermes (temp/top-p/allowed ids/penalties/...).
    RegisterOp(OpId::kApplyTemperature, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ApplyTemperatureFn>(&ApplyTemperatureKernelRocm)));
    RegisterOp(OpId::kApplyTopKTopP, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ApplyTopKTopPFn>(&ApplyTopKTopPKernelRocm)));
    RegisterOp(OpId::kComputeProbs, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ComputeProbsFn>(&ComputeProbsKernelRocm)));
    RegisterOp(OpId::kComputeLogprobs, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ComputeLogprobsFn>(&ComputeLogprobsKernelRocm)));
    RegisterOp(OpId::kRandomSample, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<RandomSampleFn>(&RandomSampleKernelRocm)));
    RegisterOp(OpId::kApplyPenalties, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ApplyPenaltiesFn>(&ApplyPenaltiesKernelRocm)));
    RegisterOp(OpId::kApplyMinP, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ApplyMinPFn>(&ApplyMinPKernelRocm)));
    RegisterOp(OpId::kApplyLogitBias, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ApplyLogitBiasFn>(&ApplyLogitBiasKernelRocm)));
    RegisterOp(OpId::kApplyAllowedTokenIds, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ApplyAllowedTokenIdsFn>(&ApplyAllowedTokenIdsKernelRocm)));
    RegisterOp(OpId::kApplyTokenMask, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ApplyTokenMaskFn>(&ApplyTokenMaskKernelRocm)));
  }
} registrar;

}  // namespace
}  // namespace vt::rocm
