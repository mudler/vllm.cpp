// Portable fused helpers used by Gemma4 (and reusable elsewhere).
// Model code MUST call these — never vt::rocm::* — so CPU/CUDA/Vulkan link.
#pragma once

#include <cstdint>

#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vt {

void RmsNormPlusAdd(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                    const Tensor& addend, const RmsNormArgs& args);

void DualRmsNormPlusRes(Queue& q, Tensor& out, const Tensor& x1, const Tensor& w1,
                        const Tensor& x2, const Tensor& w2, const Tensor& w3,
                        const Tensor& residual, const RmsNormArgs& args);

void GeluMulSeparate(Queue& q, void* out, const void* gate, const void* up, int64_t n,
                     DType dtype);

// Does `MatmulBTAlphaBeta` have an arm for this queue's device in THIS build?
// It answers the question a caller has to ask BEFORE committing to a device
// path, because the alternative is finding out from a throw: the only
// implementation in the tree is `rocm::MatmulBTAlphaBetaRocm`, so on every other
// device — and on ROCm in a build configured without `-DVLLM_CPP_HIP` — the call
// below refuses instead of computing (issue #1205).
//
// It is not a device-name test that a reader has to keep in sync by hand.
// `MatmulBTAlphaBeta` itself dispatches on this predicate, so the two cannot
// disagree: false here means the very next line throws, and a future CUDA arm
// makes both true in one edit.
bool HasMatmulBTAlphaBeta(const Queue& q);

void MatmulBTAlphaBeta(Queue& q, void* out, const void* a, const void* b, int M, int N, int K,
                       float alpha, float beta, DType dtype);

void MatmulBTFp8Channel(Queue& q, void* out, const void* a, const void* b_fp8,
                        const void* scale_bf16, int M, int N, int K, float alpha, float beta);

// Device FP8 E4M3 + BF16 channel scale → BF16 weights [N,K] (prefill hipBLAS path).
void DequantFp8ChannelBf16(Queue& q, void* out_bf16, const void* fp8, const void* scale_bf16,
                           int N, int K);

bool ExpertGeGLUBf16TopKM1(Queue& q, void* ysum, const void* x, const void* const* w_gu,
                           const void* const* w_dn, const float* wts, int G, int I, int H);

// Fused FP8 expert GeGLU top-k (T=1). Uses hipBLASLt FP8 when available, else fast HIP.
bool ExpertGeGLUFp8TopKM1(Queue& q, void* ysum, const void* x, const void* const* fp8_gu,
                          const void* const* s_gu, const void* const* fp8_dn,
                          const void* const* s_dn, const float* wts, int G, int I, int H);
bool ExpertGeGLUFp8TopKIndexed(Queue& q, void* ysum, const void* x, const void* gu_base,
                               const void* dn_base, const void* sgu_base, const void* sdn_base,
                               const int32_t* idx_dev, const float* wts_dev, int G, int I, int H);
void ApplyExpertScaleRw(Queue& q, float* rw_dev, const int32_t* ri_dev, const float* escale_dev,
                        int G, int E);
// Pre-alloc ExpertGeGLU scratch on `dev` (call after resident expert upload).
bool PrewarmExpertGeGLUFp8TopK(int dev, int G, int I, int H);

// Prefill MoE: GPU gather / weighted scatter (no host accumulation).
void MoeGatherRows(Queue& q, void* out_bf16, const void* in_bf16, const int32_t* token_ids_dev,
                   int n, int H);
void MoeWeightedScatterAdd(Queue& q, void* acc_bf16, const void* y_bf16,
                           const int32_t* token_ids_dev, const float* weights_dev, int n, int H);
void MoeZeroBf16(Queue& q, void* buf_bf16, int64_t nelem);

}  // namespace vt
