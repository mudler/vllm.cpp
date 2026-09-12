// vllm.cpp original (no upstream mirror): the LM-head output boundary.
//
// WHY THIS IS A SEAM AND NOT THREE INLINE LINES. The dtype of the LM-head
// projection's output is production behavior (issue #3116), and it is otherwise
// reachable only through a full model forward with a checkpoint. Exposing the
// projection over primitive `vt::` types lets a focused test execute the exact
// operator sequence the forward runs, over the primary's own captured head
// input, with no checkpoint and no GPU-only build. `qwen3_5_moe_block.h` is the
// same pattern for the MoE block. This is not a second path: `ForwardLayers` has
// exactly one head call site and it is this function.
#pragma once

#include <cstdint>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm {
namespace lm_head {

// out [n_out, vocab] = src [n_out, H] @ lm, returned as an owning F32 device
// buffer whose VALUES are BF16 words.
//
// `tied` selects the operator the forward selects: `vt::MatmulBT` with the
// tied embedding weight `lm` [vocab, H], or `vt::Matmul` with the untied
// lm_head weight `lm` [H, vocab] that the loader transposed at load
// (`qwen3_moe_weights.cpp`, `LoadBf16Transposed("lm_head.weight")`).
//
// THE BOUNDARY, and why it is drawn here. The pinned compiled primary stores its
// LM-head output in BF16 and widens it to F32 for the sampler
// (`.agents/specs/rocm-residual-norm.md:320`); the native forward used to store
// F32 directly, which is a measured dtype divergence of up to half a BF16 ulp
// per logit (`.agents/specs/rocm-lmhead-bf16.md`, row 0 max 9.23157e-04, row 1
// max 9.72956e-04). So the projection narrows to BF16 exactly as the primary
// does, and the SHARED `vt::CastF32` widens it back — the same op the primary's
// logits processor stands in for, and the same widening the tree already uses
// where a BF16 producer feeds an F32 consumer.
//
// THE RETURNED BUFFER STAYS F32, and that is the whole reason the widening is
// here rather than at the callers. Every consumer of the forward's logits reads
// F32: the device-logits view and its owning wrapper (`WrapDeviceLogits`,
// `ViewDeviceLogits`), the host download in `Qwen3MoeModel::Forward`, the
// captured decode graph's logits slot, and the sampler. `.agents/specs/
// rocm-residual-norm.md:325` records why: "Changing only the head buffer dtype
// would leave those views and host copies invalid."
//
// THE F32 STORE IS NOT A LEVER THIS FUNCTION OWNS. Narrowing the head output
// changes no step-6 token on either measured input: on the primary's captured
// head input both boundaries choose 118, and on the recorded native production
// logits both choose 63, where narrowing collapses the 63/118 margin to an exact
// tie that the native lowest-index tie-break resolves the same way. Hidden-state
// parity (#3115) is still required; this function mirrors a dtype and claims
// nothing more.
inline dense_attn::DBuf Project(dense_attn::Dev d, const vt::Tensor& src,
                                const vt::Tensor& lm, bool tied) {
  const int64_t n_out = src.shape[0];
  const int64_t vocab = tied ? lm.shape[0] : lm.shape[1];
  // The primary's head output dtype. `vt::Matmul`/`vt::MatmulBT` already admit a
  // BF16 store with F32 accumulation (`include/vt/ops.h:2649-2652`,
  // `src/vt/ops.cpp:125-126`), so no new op and no host round trip is involved.
  dense_attn::DBuf head(d, vt::DType::kBF16, {n_out, vocab});
  if (tied)
    vt::MatmulBT(d.q, head.t(), src, lm);
  else
    vt::Matmul(d.q, head.t(), src, lm);
  dense_attn::DBuf logits(d, vt::DType::kF32, {n_out, vocab});
  vt::CastF32(d.q, logits.t(), head.t());
  return logits;
}

}  // namespace lm_head
}  // namespace vllm
